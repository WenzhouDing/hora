# --------------------------------------------------------
# Joint Mapping Module for Cross-Hand Policy Transfer
# Provides swappable mapping strategies between different robot hands
# --------------------------------------------------------

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional


class JointMappingBase(ABC):
    """
    Abstract base class for joint mapping between two robot hands.

    Subclasses implement specific mapping strategies (e.g., normalized scaling,
    learned mapping, optimization-based mapping, etc.)
    """

    def __init__(
        self,
        source_lower_limits: torch.Tensor,
        source_upper_limits: torch.Tensor,
        target_lower_limits: torch.Tensor,
        target_upper_limits: torch.Tensor,
        source_to_target_indices: List[int],
        device: str = 'cuda:0'
    ):
        """
        Initialize the joint mapping.

        Args:
            source_lower_limits: Lower joint limits for source hand (16,)
            source_upper_limits: Upper joint limits for source hand (16,)
            target_lower_limits: Lower joint limits for target hand (16,)
            target_upper_limits: Upper joint limits for target hand (16,)
            source_to_target_indices: Index mapping from source DOF to target DOF
            device: Torch device
        """
        self.device = device
        self.source_lower = source_lower_limits.to(device)
        self.source_upper = source_upper_limits.to(device)
        self.target_lower = target_lower_limits.to(device)
        self.target_upper = target_upper_limits.to(device)
        self.source_to_target_indices = torch.tensor(
            source_to_target_indices, dtype=torch.long, device=device
        )
        # Compute inverse mapping (target to source)
        self.target_to_source_indices = torch.zeros_like(self.source_to_target_indices)
        for src_idx, tgt_idx in enumerate(source_to_target_indices):
            self.target_to_source_indices[tgt_idx] = src_idx

    @abstractmethod
    def source_to_target(self, source_joints: torch.Tensor) -> torch.Tensor:
        """
        Map joint values from source hand to target hand.

        Args:
            source_joints: Joint positions/targets in source hand space (batch, 16)

        Returns:
            target_joints: Mapped joint positions/targets for target hand (batch, 16)
        """
        pass

    @abstractmethod
    def target_to_source(self, target_joints: torch.Tensor) -> torch.Tensor:
        """
        Map joint values from target hand to source hand (inverse mapping).

        Args:
            target_joints: Joint positions/targets in target hand space (batch, 16)

        Returns:
            source_joints: Mapped joint positions/targets for source hand (batch, 16)
        """
        pass


class NormalizedScaleMapping(JointMappingBase):
    """
    Normalized scaling mapping between robot hands.

    Maps joints by:
    1. Normalizing source joints to [0, 1] range
    2. Reordering based on index mapping
    3. Scaling to target joint range

    This is a simple, interpretable mapping that preserves relative joint positions.
    """

    def __init__(
        self,
        source_lower_limits: torch.Tensor,
        source_upper_limits: torch.Tensor,
        target_lower_limits: torch.Tensor,
        target_upper_limits: torch.Tensor,
        source_to_target_indices: List[int],
        device: str = 'cuda:0'
    ):
        super().__init__(
            source_lower_limits, source_upper_limits,
            target_lower_limits, target_upper_limits,
            source_to_target_indices, device
        )
        # Precompute ranges for efficiency
        self.source_range = self.source_upper - self.source_lower + 1e-8
        self.target_range = self.target_upper - self.target_lower + 1e-8

    def source_to_target(self, source_joints: torch.Tensor) -> torch.Tensor:
        """Map from source hand to target hand using normalized scaling."""
        # Normalize source to [0, 1]
        normalized = (source_joints - self.source_lower) / self.source_range
        normalized = torch.clamp(normalized, 0.0, 1.0)

        # Reorder joints according to mapping
        normalized_reordered = normalized[:, self.source_to_target_indices]

        # Scale to target range
        target_joints = normalized_reordered * self.target_range + self.target_lower

        return target_joints

    def target_to_source(self, target_joints: torch.Tensor) -> torch.Tensor:
        """Map from target hand to source hand (inverse of source_to_target)."""
        # Normalize target to [0, 1]
        normalized = (target_joints - self.target_lower) / self.target_range
        normalized = torch.clamp(normalized, 0.0, 1.0)

        # Reorder joints according to inverse mapping
        normalized_reordered = normalized[:, self.target_to_source_indices]

        # Scale to source range
        source_joints = normalized_reordered * self.source_range + self.source_lower

        return source_joints


class IdentityMapping(JointMappingBase):
    """
    Identity mapping - just reorders joints without scaling.

    Useful for testing or when joint ranges are similar.
    """

    def source_to_target(self, source_joints: torch.Tensor) -> torch.Tensor:
        """Direct reordering without scaling."""
        return source_joints[:, self.source_to_target_indices]

    def target_to_source(self, target_joints: torch.Tensor) -> torch.Tensor:
        """Inverse reordering without scaling."""
        return target_joints[:, self.target_to_source_indices]


class FingertipIKMapping(JointMappingBase):
    """
    Inverse Kinematics based mapping using simplified finger kinematics.

    For each finger, computes fingertip position using forward kinematics,
    then uses iterative IK to find target joint angles that achieve
    similar fingertip positions.

    Uses simplified 2D planar kinematics per finger (ignoring spread).
    """

    def __init__(
        self,
        source_lower_limits: torch.Tensor,
        source_upper_limits: torch.Tensor,
        target_lower_limits: torch.Tensor,
        target_upper_limits: torch.Tensor,
        source_to_target_indices: List[int],
        device: str = 'cuda:0',
        source_link_lengths: Optional[List[List[float]]] = None,
        target_link_lengths: Optional[List[List[float]]] = None,
        ik_iterations: int = 10,
        ik_step_size: float = 0.5
    ):
        super().__init__(
            source_lower_limits, source_upper_limits,
            target_lower_limits, target_upper_limits,
            source_to_target_indices, device
        )

        # Default link lengths (approximate, in meters)
        # [proximal, middle, distal] for each finger
        # Allegro hand approximate link lengths
        self.source_links = source_link_lengths or [
            [0.054, 0.038, 0.044],  # Index
            [0.054, 0.038, 0.044],  # Thumb (simplified)
            [0.054, 0.038, 0.044],  # Middle
            [0.054, 0.038, 0.044],  # Ring
        ]
        # LEAP hand approximate link lengths
        self.target_links = target_link_lengths or [
            [0.050, 0.032, 0.032],  # Index
            [0.050, 0.032, 0.032],  # Thumb (simplified)
            [0.050, 0.032, 0.032],  # Middle
            [0.050, 0.032, 0.032],  # Ring
        ]

        self.source_links = torch.tensor(self.source_links, device=device, dtype=torch.float)
        self.target_links = torch.tensor(self.target_links, device=device, dtype=torch.float)

        self.ik_iterations = ik_iterations
        self.ik_step_size = ik_step_size

        # Precompute ranges
        self.source_range = self.source_upper - self.source_lower + 1e-8
        self.target_range = self.target_upper - self.target_lower + 1e-8

    def _forward_kinematics_2d(self, joints: torch.Tensor, link_lengths: torch.Tensor, finger_idx: int) -> torch.Tensor:
        """
        Simplified 2D forward kinematics for a finger.

        Args:
            joints: Joint angles for one finger (batch, 4) - [spread, mcp, pip, dip]
            link_lengths: Link lengths (3,) - [proximal, middle, distal]
            finger_idx: Finger index (0-3)

        Returns:
            fingertip_pos: 2D position (batch, 2) - [x, y] in finger plane
        """
        batch_size = joints.shape[0]

        # For simplicity, ignore spread (joint 0) and use joints 1,2,3 for 2D planar IK
        # Cumulative angles
        theta1 = joints[:, 1]  # MCP
        theta2 = theta1 + joints[:, 2]  # MCP + PIP
        theta3 = theta2 + joints[:, 3]  # MCP + PIP + DIP

        # Forward kinematics
        l1, l2, l3 = link_lengths[0], link_lengths[1], link_lengths[2]

        x = l1 * torch.cos(theta1) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)
        y = l1 * torch.sin(theta1) + l2 * torch.sin(theta2) + l3 * torch.sin(theta3)

        return torch.stack([x, y], dim=-1)

    def _inverse_kinematics_2d(
        self,
        target_pos: torch.Tensor,
        link_lengths: torch.Tensor,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        initial_joints: torch.Tensor
    ) -> torch.Tensor:
        """
        Iterative IK using Jacobian transpose method.

        Args:
            target_pos: Target fingertip position (batch, 2)
            link_lengths: Link lengths (3,)
            joint_lower: Lower joint limits (4,)
            joint_upper: Upper joint limits (4,)
            initial_joints: Initial joint guess (batch, 4)

        Returns:
            joints: Solved joint angles (batch, 4)
        """
        joints = initial_joints.clone()
        batch_size = joints.shape[0]

        for _ in range(self.ik_iterations):
            # Current fingertip position
            current_pos = self._forward_kinematics_2d(joints, link_lengths, 0)

            # Position error
            error = target_pos - current_pos

            # Compute Jacobian (simplified - just for joints 1,2,3)
            theta1 = joints[:, 1]
            theta2 = theta1 + joints[:, 2]
            theta3 = theta2 + joints[:, 3]

            l1, l2, l3 = link_lengths[0], link_lengths[1], link_lengths[2]

            # Jacobian columns
            j1_x = -l1 * torch.sin(theta1) - l2 * torch.sin(theta2) - l3 * torch.sin(theta3)
            j1_y = l1 * torch.cos(theta1) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)

            j2_x = -l2 * torch.sin(theta2) - l3 * torch.sin(theta3)
            j2_y = l2 * torch.cos(theta2) + l3 * torch.cos(theta3)

            j3_x = -l3 * torch.sin(theta3)
            j3_y = l3 * torch.cos(theta3)

            # Jacobian transpose method: delta_q = alpha * J^T * error
            delta_q1 = self.ik_step_size * (j1_x * error[:, 0] + j1_y * error[:, 1])
            delta_q2 = self.ik_step_size * (j2_x * error[:, 0] + j2_y * error[:, 1])
            delta_q3 = self.ik_step_size * (j3_x * error[:, 0] + j3_y * error[:, 1])

            # Update joints (skip spread joint 0)
            joints[:, 1] = joints[:, 1] + delta_q1
            joints[:, 2] = joints[:, 2] + delta_q2
            joints[:, 3] = joints[:, 3] + delta_q3

            # Clamp to limits
            joints = torch.clamp(joints, joint_lower, joint_upper)

        return joints

    def source_to_target(self, source_joints: torch.Tensor) -> torch.Tensor:
        """Map using IK to match fingertip positions."""
        batch_size = source_joints.shape[0]

        # Reorder source joints
        source_reordered = source_joints[:, self.source_to_target_indices]

        # Initialize target with normalized scale mapping as starting point
        normalized = (source_joints - self.source_lower) / self.source_range
        normalized = torch.clamp(normalized, 0.0, 1.0)
        normalized_reordered = normalized[:, self.source_to_target_indices]
        target_joints = normalized_reordered * self.target_range + self.target_lower

        # For each finger, apply IK correction
        for finger_idx in range(4):
            start_idx = finger_idx * 4
            end_idx = start_idx + 4

            # Get source finger joints and compute target fingertip position
            source_finger = source_reordered[:, start_idx:end_idx]
            target_fingertip = self._forward_kinematics_2d(
                source_finger, self.source_links[finger_idx], finger_idx
            )

            # No scaling - use fingertip position directly
            # The IK will find joint angles that reach the same position

            # Run IK for target finger
            target_finger_lower = self.target_lower[start_idx:end_idx]
            target_finger_upper = self.target_upper[start_idx:end_idx]
            initial_finger = target_joints[:, start_idx:end_idx]

            solved_finger = self._inverse_kinematics_2d(
                target_fingertip,
                self.target_links[finger_idx],
                target_finger_lower,
                target_finger_upper,
                initial_finger
            )

            # Copy spread joint directly (scaled)
            solved_finger[:, 0] = initial_finger[:, 0]

            target_joints[:, start_idx:end_idx] = solved_finger

        return target_joints

    def target_to_source(self, target_joints: torch.Tensor) -> torch.Tensor:
        """Inverse mapping using IK."""
        batch_size = target_joints.shape[0]

        # Initialize with normalized scale mapping
        normalized = (target_joints - self.target_lower) / self.target_range
        normalized = torch.clamp(normalized, 0.0, 1.0)
        normalized_reordered = normalized[:, self.target_to_source_indices]
        source_joints = normalized_reordered * self.source_range + self.source_lower

        # Reorder target joints for FK
        target_reordered = target_joints[:, self.target_to_source_indices]

        # For each finger, apply IK correction
        for finger_idx in range(4):
            start_idx = finger_idx * 4
            end_idx = start_idx + 4

            # Get target finger joints and compute source fingertip position
            target_finger = target_reordered[:, start_idx:end_idx]
            source_fingertip = self._forward_kinematics_2d(
                target_finger, self.target_links[finger_idx], finger_idx
            )

            # No scaling - use fingertip position directly

            # Run IK for source finger
            source_finger_lower = self.source_lower[start_idx:end_idx]
            source_finger_upper = self.source_upper[start_idx:end_idx]
            initial_finger = source_joints[:, start_idx:end_idx]

            solved_finger = self._inverse_kinematics_2d(
                source_fingertip,
                self.source_links[finger_idx],
                source_finger_lower,
                source_finger_upper,
                initial_finger
            )

            # Copy spread joint directly (scaled)
            solved_finger[:, 0] = initial_finger[:, 0]

            source_joints[:, start_idx:end_idx] = solved_finger

        return source_joints


class FingertipIK3DMapping(JointMappingBase):
    """
    Full 3D Inverse Kinematics based mapping.

    For each finger, computes 3D fingertip position using forward kinematics
    that properly accounts for the spread joint, then uses Jacobian-based IK
    (damped least squares) to solve for target joint angles.

    Coordinate system per finger:
    - X: along finger when extended (distal direction)
    - Y: perpendicular to palm (flexion direction)
    - Z: lateral (spread/abduction direction)

    Joint order: [spread, MCP, PIP, DIP]
    - Spread: rotates around Y axis
    - MCP, PIP, DIP: rotate around Z axis (in spread-rotated frame)
    """

    def __init__(
        self,
        source_lower_limits: torch.Tensor,
        source_upper_limits: torch.Tensor,
        target_lower_limits: torch.Tensor,
        target_upper_limits: torch.Tensor,
        source_to_target_indices: List[int],
        device: str = 'cuda:0',
        source_link_lengths: Optional[List[List[float]]] = None,
        target_link_lengths: Optional[List[List[float]]] = None,
        ik_iterations: int = 5,  # Reduced for performance
        damping: float = 0.05  # Increased for stability
    ):
        super().__init__(
            source_lower_limits, source_upper_limits,
            target_lower_limits, target_upper_limits,
            source_to_target_indices, device
        )

        # Default link lengths (approximate, in meters)
        # [proximal, middle, distal] for each finger
        # Allegro hand approximate link lengths
        self.source_links = source_link_lengths or [
            [0.054, 0.038, 0.044],  # Index
            [0.054, 0.038, 0.044],  # Thumb
            [0.054, 0.038, 0.044],  # Middle
            [0.054, 0.038, 0.044],  # Ring
        ]
        # LEAP hand approximate link lengths
        self.target_links = target_link_lengths or [
            [0.050, 0.032, 0.032],  # Index
            [0.050, 0.032, 0.032],  # Thumb
            [0.050, 0.032, 0.032],  # Middle
            [0.050, 0.032, 0.032],  # Ring
        ]

        self.source_links = torch.tensor(self.source_links, device=device, dtype=torch.float)
        self.target_links = torch.tensor(self.target_links, device=device, dtype=torch.float)

        self.ik_iterations = ik_iterations
        self.damping = damping

        # Precompute ranges for normalized scale fallback
        self.source_range = self.source_upper - self.source_lower + 1e-8
        self.target_range = self.target_upper - self.target_lower + 1e-8

    def _forward_kinematics_3d(
        self, joints: torch.Tensor, link_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Full 3D forward kinematics for a finger.

        Args:
            joints: Joint angles for one finger (batch, 4) - [spread, mcp, pip, dip]
            link_lengths: Link lengths (3,) - [proximal, middle, distal]

        Returns:
            fingertip_pos: 3D position (batch, 3) - [x, y, z]
        """
        spread = joints[:, 0]
        mcp = joints[:, 1]
        pip = joints[:, 2]
        dip = joints[:, 3]

        # Cumulative flexion angles
        theta1 = mcp
        theta2 = mcp + pip
        theta3 = mcp + pip + dip

        l1, l2, l3 = link_lengths[0], link_lengths[1], link_lengths[2]

        # Position in finger plane (before spread rotation)
        # x_plane: along finger direction
        # y_plane: perpendicular (flexion direction)
        x_plane = l1 * torch.cos(theta1) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)
        y_plane = l1 * torch.sin(theta1) + l2 * torch.sin(theta2) + l3 * torch.sin(theta3)

        # Apply spread rotation (around Y axis)
        # x = x_plane * cos(spread)
        # y = y_plane (unchanged)
        # z = x_plane * sin(spread)
        x = x_plane * torch.cos(spread)
        y = y_plane
        z = x_plane * torch.sin(spread)

        return torch.stack([x, y, z], dim=-1)

    def _jacobian_3d(
        self, joints: torch.Tensor, link_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 3x4 Jacobian matrix for a finger.

        Args:
            joints: Joint angles (batch, 4) - [spread, mcp, pip, dip]
            link_lengths: Link lengths (3,)

        Returns:
            jacobian: Jacobian matrix (batch, 3, 4)
        """
        batch_size = joints.shape[0]
        spread = joints[:, 0]
        mcp = joints[:, 1]
        pip = joints[:, 2]
        dip = joints[:, 3]

        theta1 = mcp
        theta2 = mcp + pip
        theta3 = mcp + pip + dip

        l1, l2, l3 = link_lengths[0], link_lengths[1], link_lengths[2]

        # r = reach in finger plane (x component before spread)
        r = l1 * torch.cos(theta1) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)

        # Derivatives of r w.r.t. flexion joints
        dr_dmcp = -l1 * torch.sin(theta1) - l2 * torch.sin(theta2) - l3 * torch.sin(theta3)
        dr_dpip = -l2 * torch.sin(theta2) - l3 * torch.sin(theta3)
        dr_ddip = -l3 * torch.sin(theta3)

        # Derivatives of y w.r.t. flexion joints
        dy_dmcp = l1 * torch.cos(theta1) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)
        dy_dpip = l2 * torch.cos(theta2) + l3 * torch.cos(theta3)
        dy_ddip = l3 * torch.cos(theta3)

        cos_s = torch.cos(spread)
        sin_s = torch.sin(spread)

        # Jacobian: d[x,y,z]/d[spread, mcp, pip, dip]
        # dx/d_spread = -r * sin(spread)
        # dx/d_mcp = dr_dmcp * cos(spread)
        # dx/d_pip = dr_dpip * cos(spread)
        # dx/d_dip = dr_ddip * cos(spread)

        # dy/d_spread = 0
        # dy/d_mcp = dy_dmcp
        # dy/d_pip = dy_dpip
        # dy/d_dip = dy_ddip

        # dz/d_spread = r * cos(spread)
        # dz/d_mcp = dr_dmcp * sin(spread)
        # dz/d_pip = dr_dpip * sin(spread)
        # dz/d_dip = dr_ddip * sin(spread)

        J = torch.zeros(batch_size, 3, 4, device=joints.device)

        # Row 0: dx/d[spread, mcp, pip, dip]
        J[:, 0, 0] = -r * sin_s
        J[:, 0, 1] = dr_dmcp * cos_s
        J[:, 0, 2] = dr_dpip * cos_s
        J[:, 0, 3] = dr_ddip * cos_s

        # Row 1: dy/d[spread, mcp, pip, dip]
        J[:, 1, 0] = 0
        J[:, 1, 1] = dy_dmcp
        J[:, 1, 2] = dy_dpip
        J[:, 1, 3] = dy_ddip

        # Row 2: dz/d[spread, mcp, pip, dip]
        J[:, 2, 0] = r * cos_s
        J[:, 2, 1] = dr_dmcp * sin_s
        J[:, 2, 2] = dr_dpip * sin_s
        J[:, 2, 3] = dr_ddip * sin_s

        return J

    def _inverse_kinematics_3d(
        self,
        target_pos: torch.Tensor,
        link_lengths: torch.Tensor,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        initial_joints: torch.Tensor
    ) -> torch.Tensor:
        """
        Iterative 3D IK using damped least squares (Levenberg-Marquardt).

        Args:
            target_pos: Target 3D fingertip position (batch, 3)
            link_lengths: Link lengths (3,)
            joint_lower: Lower joint limits (4,)
            joint_upper: Upper joint limits (4,)
            initial_joints: Initial joint guess (batch, 4)

        Returns:
            joints: Solved joint angles (batch, 4)
        """
        joints = initial_joints.clone()
        batch_size = joints.shape[0]

        # Pre-allocate damping matrix
        damping_matrix = self.damping * torch.eye(4, device=joints.device).unsqueeze(0).expand(batch_size, -1, -1)

        for _ in range(self.ik_iterations):
            # Current fingertip position
            current_pos = self._forward_kinematics_3d(joints, link_lengths)

            # Position error (batch, 3)
            error = target_pos - current_pos

            # Early termination if error is small enough (< 1mm)
            error_norm = torch.norm(error, dim=-1)
            if error_norm.max() < 0.001:
                break

            # Compute Jacobian (batch, 3, 4)
            J = self._jacobian_3d(joints, link_lengths)

            # Damped least squares: delta_q = (J^T J + λI)^-1 J^T error
            # J^T @ J + λI (batch, 4, 4)
            JtJ = torch.bmm(J.transpose(1, 2), J)
            JtJ_damped = JtJ + damping_matrix

            # J^T @ error (batch, 4)
            Jt_error = torch.bmm(J.transpose(1, 2), error.unsqueeze(-1)).squeeze(-1)

            # Solve for delta_q
            delta_q = torch.linalg.solve(JtJ_damped, Jt_error)

            # Update joints with step size limiting for stability
            delta_q = torch.clamp(delta_q, -0.5, 0.5)  # Limit step size
            joints = joints + delta_q

            # Clamp to limits
            joints = torch.clamp(joints, joint_lower, joint_upper)

        return joints

    def source_to_target(self, source_joints: torch.Tensor) -> torch.Tensor:
        """Map using 3D IK to match fingertip positions."""
        batch_size = source_joints.shape[0]

        # Reorder source joints to target order
        source_reordered = source_joints[:, self.source_to_target_indices]

        # Initialize target with normalized scale mapping as starting point
        normalized = (source_joints - self.source_lower) / self.source_range
        normalized = torch.clamp(normalized, 0.0, 1.0)
        normalized_reordered = normalized[:, self.source_to_target_indices]
        target_joints = normalized_reordered * self.target_range + self.target_lower

        # For each finger, apply 3D IK
        for finger_idx in range(4):
            start_idx = finger_idx * 4
            end_idx = start_idx + 4

            # Get source finger joints and compute 3D fingertip position
            source_finger = source_reordered[:, start_idx:end_idx]
            target_fingertip = self._forward_kinematics_3d(
                source_finger, self.source_links[finger_idx]
            )

            # No scaling - use fingertip position directly

            # Run 3D IK for target finger
            target_finger_lower = self.target_lower[start_idx:end_idx]
            target_finger_upper = self.target_upper[start_idx:end_idx]
            initial_finger = target_joints[:, start_idx:end_idx]

            solved_finger = self._inverse_kinematics_3d(
                target_fingertip,
                self.target_links[finger_idx],
                target_finger_lower,
                target_finger_upper,
                initial_finger
            )

            target_joints[:, start_idx:end_idx] = solved_finger

        return target_joints

    def target_to_source(self, target_joints: torch.Tensor) -> torch.Tensor:
        """Inverse mapping using 3D IK."""
        batch_size = target_joints.shape[0]

        # Initialize with normalized scale mapping
        normalized = (target_joints - self.target_lower) / self.target_range
        normalized = torch.clamp(normalized, 0.0, 1.0)
        normalized_reordered = normalized[:, self.target_to_source_indices]
        source_joints = normalized_reordered * self.source_range + self.source_lower

        # Reorder target joints
        target_reordered = target_joints[:, self.target_to_source_indices]

        # For each finger, apply 3D IK
        for finger_idx in range(4):
            start_idx = finger_idx * 4
            end_idx = start_idx + 4

            # Get target finger joints and compute 3D fingertip position
            target_finger = target_reordered[:, start_idx:end_idx]
            source_fingertip = self._forward_kinematics_3d(
                target_finger, self.target_links[finger_idx]
            )

            # No scaling - use fingertip position directly

            # Run 3D IK for source finger
            source_finger_lower = self.source_lower[start_idx:end_idx]
            source_finger_upper = self.source_upper[start_idx:end_idx]
            initial_finger = source_joints[:, start_idx:end_idx]

            solved_finger = self._inverse_kinematics_3d(
                source_fingertip,
                self.source_links[finger_idx],
                source_finger_lower,
                source_finger_upper,
                initial_finger
            )

            source_joints[:, start_idx:end_idx] = solved_finger

        return source_joints

    def sanity_check(self, num_samples: int = 100, verbose: bool = True) -> dict:
        """
        Sanity check: verify FK and IK are consistent.

        Tests:
        1. FK(random_joints) -> position -> IK(position) should recover joints
        2. Position error after IK should be small
        3. Round-trip mapping should approximately preserve fingertip positions

        Returns:
            dict with error statistics
        """
        results = {}

        # Sample random joints within limits
        source_joints = torch.rand(num_samples, 16, device=self.device)
        source_joints = source_joints * self.source_range + self.source_lower

        if verbose:
            print("=" * 60)
            print("3D IK Mapping Sanity Check")
            print("=" * 60)

        # Test 1: FK consistency per finger
        fk_errors = []
        ik_errors = []

        for finger_idx in range(4):
            start_idx = finger_idx * 4
            end_idx = start_idx + 4

            finger_joints = source_joints[:, start_idx:end_idx]
            link_lengths = self.source_links[finger_idx]
            joint_lower = self.source_lower[start_idx:end_idx]
            joint_upper = self.source_upper[start_idx:end_idx]

            # FK
            positions = self._forward_kinematics_3d(finger_joints, link_lengths)

            # IK to recover joints
            recovered_joints = self._inverse_kinematics_3d(
                positions, link_lengths, joint_lower, joint_upper, finger_joints.clone()
            )

            # FK again to check position
            recovered_positions = self._forward_kinematics_3d(recovered_joints, link_lengths)

            # Position error
            pos_error = torch.norm(positions - recovered_positions, dim=-1).mean().item()
            fk_errors.append(pos_error)

            # Joint error (may not be exact due to redundancy)
            joint_error = torch.abs(finger_joints - recovered_joints).mean().item()
            ik_errors.append(joint_error)

            if verbose:
                print(f"Finger {finger_idx}: FK pos error = {pos_error:.6f}m, joint error = {joint_error:.4f}rad")

        results['fk_position_errors'] = fk_errors
        results['ik_joint_errors'] = ik_errors
        results['mean_fk_error'] = np.mean(fk_errors)
        results['mean_ik_error'] = np.mean(ik_errors)

        # Test 2: Full mapping round-trip
        target_joints = self.source_to_target(source_joints)
        recovered_source = self.target_to_source(target_joints)

        # Compare fingertip positions
        source_reordered = source_joints[:, self.source_to_target_indices]
        recovered_reordered = recovered_source[:, self.source_to_target_indices]

        fingertip_errors = []
        for finger_idx in range(4):
            start_idx = finger_idx * 4
            end_idx = start_idx + 4

            source_finger = source_reordered[:, start_idx:end_idx]
            recovered_finger = recovered_reordered[:, start_idx:end_idx]

            source_pos = self._forward_kinematics_3d(source_finger, self.source_links[finger_idx])
            recovered_pos = self._forward_kinematics_3d(recovered_finger, self.source_links[finger_idx])

            error = torch.norm(source_pos - recovered_pos, dim=-1).mean().item()
            fingertip_errors.append(error)

        results['roundtrip_fingertip_errors'] = fingertip_errors
        results['mean_roundtrip_error'] = np.mean(fingertip_errors)

        if verbose:
            print("-" * 60)
            print(f"Mean FK position error: {results['mean_fk_error']:.6f}m")
            print(f"Mean IK joint error: {results['mean_ik_error']:.4f}rad")
            print(f"Mean round-trip fingertip error: {results['mean_roundtrip_error']:.6f}m")
            print("=" * 60)

            # Pass/fail
            if results['mean_fk_error'] < 0.001:  # < 1mm
                print("✓ FK/IK consistency: PASS")
            else:
                print("✗ FK/IK consistency: FAIL (error > 1mm)")

            if results['mean_roundtrip_error'] < 0.005:  # < 5mm
                print("✓ Round-trip mapping: PASS")
            else:
                print("✗ Round-trip mapping: FAIL (error > 5mm)")

        return results


class CalibratedPolynomialMapping(JointMappingBase):
    """
    Calibration-based mapping using polynomial features.

    Offline calibration samples source joint space, computes fingertip positions
    via FK, solves IK for target joints, then fits a polynomial mapping.

    Runtime: Apply learned polynomial transform (fast matrix multiply).

    Polynomial features: [1, x1, x2, ..., x16, x1*x1, x1*x2, ..., x16*x16]
    This gives 1 + 16 + 136 = 153 features for degree 2.
    """

    def __init__(
        self,
        source_lower_limits: torch.Tensor,
        source_upper_limits: torch.Tensor,
        target_lower_limits: torch.Tensor,
        target_upper_limits: torch.Tensor,
        source_to_target_indices: List[int],
        device: str = 'cuda:0',
        calibration_file: Optional[str] = None,
        degree: int = 2
    ):
        super().__init__(
            source_lower_limits, source_upper_limits,
            target_lower_limits, target_upper_limits,
            source_to_target_indices, device
        )

        self.degree = degree
        self.n_joints = 16

        # Compute number of polynomial features
        # degree=2: 1 (bias) + 16 (linear) + 16*17/2 (quadratic) = 153
        self.n_features = self._compute_n_features()

        # Initialize weights (will be loaded from calibration file)
        self.forward_weights = None  # (n_features, 16)
        self.inverse_weights = None  # (n_features, 16)

        if calibration_file is not None:
            self.load_calibration(calibration_file)

    def _compute_n_features(self) -> int:
        """Compute number of polynomial features for given degree."""
        n = self.n_joints
        if self.degree == 1:
            return 1 + n  # bias + linear
        elif self.degree == 2:
            return 1 + n + (n * (n + 1)) // 2  # bias + linear + quadratic
        else:
            raise ValueError(f"Degree {self.degree} not supported, use 1 or 2")

    def _polynomial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute polynomial features up to specified degree.

        Args:
            x: Input tensor (batch, 16)

        Returns:
            features: Polynomial features (batch, n_features)
        """
        batch_size = x.shape[0]
        features = [torch.ones(batch_size, 1, device=x.device)]  # bias
        features.append(x)  # linear terms

        if self.degree >= 2:
            # Quadratic terms (upper triangular including diagonal)
            quad_terms = []
            for i in range(self.n_joints):
                for j in range(i, self.n_joints):
                    quad_terms.append(x[:, i:i+1] * x[:, j:j+1])
            features.append(torch.cat(quad_terms, dim=1))

        return torch.cat(features, dim=1)

    def _normalize_joints(self, joints: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Normalize joints to [-1, 1] range for better numerical stability."""
        return 2.0 * (joints - lower) / (upper - lower + 1e-8) - 1.0

    def _denormalize_joints(self, normalized: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Denormalize joints from [-1, 1] back to original range."""
        return (normalized + 1.0) * 0.5 * (upper - lower) + lower

    def load_calibration(self, calibration_file: str):
        """Load pre-computed calibration weights."""
        data = np.load(calibration_file)
        self.forward_weights = torch.tensor(data['forward_weights'], device=self.device, dtype=torch.float32)
        self.inverse_weights = torch.tensor(data['inverse_weights'], device=self.device, dtype=torch.float32)
        print(f"Loaded calibration from {calibration_file}")
        print(f"  Forward weights: {self.forward_weights.shape}")
        print(f"  Inverse weights: {self.inverse_weights.shape}")

    def source_to_target(self, source_joints: torch.Tensor) -> torch.Tensor:
        """Map from source to target using calibrated polynomial transform."""
        if self.forward_weights is None:
            raise RuntimeError("Calibration not loaded. Call load_calibration() first.")

        # Normalize source joints
        normalized = self._normalize_joints(source_joints, self.source_lower, self.source_upper)

        # Compute polynomial features
        features = self._polynomial_features(normalized)

        # Apply learned mapping
        target_normalized = features @ self.forward_weights

        # Denormalize and clamp to target limits
        target_joints = self._denormalize_joints(target_normalized, self.target_lower, self.target_upper)
        target_joints = torch.clamp(target_joints, self.target_lower, self.target_upper)

        return target_joints

    def target_to_source(self, target_joints: torch.Tensor) -> torch.Tensor:
        """Map from target to source using calibrated polynomial transform."""
        if self.inverse_weights is None:
            raise RuntimeError("Calibration not loaded. Call load_calibration() first.")

        # Normalize target joints
        normalized = self._normalize_joints(target_joints, self.target_lower, self.target_upper)

        # Compute polynomial features
        features = self._polynomial_features(normalized)

        # Apply learned mapping
        source_normalized = features @ self.inverse_weights

        # Denormalize and clamp to source limits
        source_joints = self._denormalize_joints(source_normalized, self.source_lower, self.source_upper)
        source_joints = torch.clamp(source_joints, self.source_lower, self.source_upper)

        return source_joints


def calibrate_polynomial_mapping(
    source_fk_func,
    target_ik_func,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    num_samples: int = 50000,
    degree: int = 2,
    regularization: float = 1e-4,
    verbose: bool = True
) -> dict:
    """
    Generate calibration data by sampling source joints and solving IK.

    Args:
        source_fk_func: Forward kinematics function source_joints -> fingertip_positions
        target_ik_func: Inverse kinematics function fingertip_positions -> target_joints (or None)
        source_lower: Source hand lower joint limits (16,)
        source_upper: Source hand upper joint limits (16,)
        target_lower: Target hand lower joint limits (16,)
        target_upper: Target hand upper joint limits (16,)
        num_samples: Number of random samples for calibration
        degree: Polynomial degree (1 or 2)
        regularization: Ridge regression regularization strength
        verbose: Print progress

    Returns:
        Dictionary with 'forward_weights' and 'inverse_weights' arrays
    """
    n_joints = 16

    # Compute number of polynomial features
    if degree == 1:
        n_features = 1 + n_joints
    elif degree == 2:
        n_features = 1 + n_joints + (n_joints * (n_joints + 1)) // 2
    else:
        raise ValueError(f"Degree {degree} not supported")

    def compute_poly_features(x: np.ndarray) -> np.ndarray:
        """Compute polynomial features for numpy array."""
        batch = x.shape[0]
        features = [np.ones((batch, 1))]  # bias
        features.append(x)  # linear
        if degree >= 2:
            quad = []
            for i in range(n_joints):
                for j in range(i, n_joints):
                    quad.append(x[:, i:i+1] * x[:, j:j+1])
            features.append(np.concatenate(quad, axis=1))
        return np.concatenate(features, axis=1)

    def normalize(joints, lower, upper):
        return 2.0 * (joints - lower) / (upper - lower + 1e-8) - 1.0

    def denormalize(normalized, lower, upper):
        return (normalized + 1.0) * 0.5 * (upper - lower) + lower

    if verbose:
        print(f"Calibrating polynomial mapping (degree={degree}, n_features={n_features})")
        print(f"Sampling {num_samples} joint configurations...")

    # Sample random source joint configurations
    source_samples = np.random.uniform(source_lower, source_upper, size=(num_samples, n_joints))

    # Compute target joints via FK + IK
    if target_ik_func is not None:
        # Use provided IK function
        fingertip_positions = source_fk_func(source_samples)
        target_samples = target_ik_func(fingertip_positions)
    else:
        # Fallback: use normalized scaling as proxy for "ground truth"
        # This is useful when we don't have a true IK solver
        source_norm = normalize(source_samples, source_lower, source_upper)
        target_samples = denormalize(source_norm, target_lower, target_upper)

    if verbose:
        print(f"Collected {len(source_samples)} sample pairs")

    # Normalize all samples
    source_norm = normalize(source_samples, source_lower, source_upper)
    target_norm = normalize(target_samples, target_lower, target_upper)

    # Compute polynomial features
    source_features = compute_poly_features(source_norm)
    target_features = compute_poly_features(target_norm)

    if verbose:
        print(f"Feature matrix shape: {source_features.shape}")
        print("Fitting forward mapping (source -> target)...")

    # Fit forward mapping: target_norm = source_features @ W_forward
    # Ridge regression: W = (X^T X + λI)^-1 X^T Y
    XtX = source_features.T @ source_features
    XtY = source_features.T @ target_norm
    W_forward = np.linalg.solve(
        XtX + regularization * np.eye(n_features),
        XtY
    )

    # Compute forward error
    pred_target = source_features @ W_forward
    forward_mse = np.mean((pred_target - target_norm) ** 2)

    if verbose:
        print(f"  Forward MSE (normalized): {forward_mse:.6f}")
        print("Fitting inverse mapping (target -> source)...")

    # Fit inverse mapping: source_norm = target_features @ W_inverse
    XtX_inv = target_features.T @ target_features
    XtY_inv = target_features.T @ source_norm
    W_inverse = np.linalg.solve(
        XtX_inv + regularization * np.eye(n_features),
        XtY_inv
    )

    # Compute inverse error
    pred_source = target_features @ W_inverse
    inverse_mse = np.mean((pred_source - source_norm) ** 2)

    if verbose:
        print(f"  Inverse MSE (normalized): {inverse_mse:.6f}")
        print("Calibration complete!")

    return {
        'forward_weights': W_forward.astype(np.float32),
        'inverse_weights': W_inverse.astype(np.float32),
        'forward_mse': forward_mse,
        'inverse_mse': inverse_mse,
        'degree': degree,
        'num_samples': num_samples
    }


# Pre-defined index mappings for common hand pairs
# These account for different URDF tree traversal orders

# Allegro to LEAP mapping
# Both hands: DOFs 0-3 (Index), 4-7 (Thumb), 8-11 (Middle), 12-15 (Ring)
# Within Index/Middle/Ring: Allegro has spread first, LEAP has MCP first
# Thumb: Same ordering in both hands
ALLEGRO_TO_LEAP_INDICES = [1, 0, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]

# LEAP to Allegro mapping (inverse of above)
LEAP_TO_ALLEGRO_INDICES = [1, 0, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]


def create_allegro_to_leap_mapping(
    allegro_lower: torch.Tensor,
    allegro_upper: torch.Tensor,
    leap_lower: torch.Tensor,
    leap_upper: torch.Tensor,
    mapping_type: str = 'normalized_scale',
    device: str = 'cuda:0',
    calibration_file: Optional[str] = None
) -> JointMappingBase:
    """
    Factory function to create Allegro-to-LEAP joint mapping.

    Args:
        allegro_lower: Allegro hand lower joint limits
        allegro_upper: Allegro hand upper joint limits
        leap_lower: LEAP hand lower joint limits
        leap_upper: LEAP hand upper joint limits
        mapping_type: Type of mapping ('normalized_scale', 'identity', 'fingertip_ik', 'calibrated')
        device: Torch device
        calibration_file: Path to calibration file (required for 'calibrated' type)

    Returns:
        JointMappingBase instance
    """
    if mapping_type == 'normalized_scale':
        return NormalizedScaleMapping(
            allegro_lower, allegro_upper,
            leap_lower, leap_upper,
            ALLEGRO_TO_LEAP_INDICES,
            device
        )
    elif mapping_type == 'identity':
        return IdentityMapping(
            allegro_lower, allegro_upper,
            leap_lower, leap_upper,
            ALLEGRO_TO_LEAP_INDICES,
            device
        )
    elif mapping_type == 'fingertip_ik':
        return FingertipIKMapping(
            allegro_lower, allegro_upper,
            leap_lower, leap_upper,
            ALLEGRO_TO_LEAP_INDICES,
            device
        )
    elif mapping_type == 'fingertip_ik_3d':
        return FingertipIK3DMapping(
            allegro_lower, allegro_upper,
            leap_lower, leap_upper,
            ALLEGRO_TO_LEAP_INDICES,
            device
        )
    elif mapping_type == 'calibrated':
        if calibration_file is None:
            calibration_file = 'cache/allegro_to_leap_calibration.npz'
        return CalibratedPolynomialMapping(
            allegro_lower, allegro_upper,
            leap_lower, leap_upper,
            ALLEGRO_TO_LEAP_INDICES,
            device,
            calibration_file=calibration_file
        )
    else:
        raise ValueError(f"Unknown mapping type: {mapping_type}")


def create_leap_to_allegro_mapping(
    leap_lower: torch.Tensor,
    leap_upper: torch.Tensor,
    allegro_lower: torch.Tensor,
    allegro_upper: torch.Tensor,
    mapping_type: str = 'normalized_scale',
    device: str = 'cuda:0',
    calibration_file: Optional[str] = None
) -> JointMappingBase:
    """
    Factory function to create LEAP-to-Allegro joint mapping.

    Args:
        leap_lower: LEAP hand lower joint limits
        leap_upper: LEAP hand upper joint limits
        allegro_lower: Allegro hand lower joint limits
        allegro_upper: Allegro hand upper joint limits
        mapping_type: Type of mapping ('normalized_scale', 'identity', 'fingertip_ik', 'calibrated')
        device: Torch device
        calibration_file: Path to calibration file (required for 'calibrated' type)

    Returns:
        JointMappingBase instance
    """
    if mapping_type == 'normalized_scale':
        return NormalizedScaleMapping(
            leap_lower, leap_upper,
            allegro_lower, allegro_upper,
            LEAP_TO_ALLEGRO_INDICES,
            device
        )
    elif mapping_type == 'identity':
        return IdentityMapping(
            leap_lower, leap_upper,
            allegro_lower, allegro_upper,
            LEAP_TO_ALLEGRO_INDICES,
            device
        )
    elif mapping_type == 'fingertip_ik':
        # Swap link lengths for inverse direction
        return FingertipIKMapping(
            leap_lower, leap_upper,
            allegro_lower, allegro_upper,
            LEAP_TO_ALLEGRO_INDICES,
            device,
            source_link_lengths=[
                [0.050, 0.032, 0.032],  # LEAP Index
                [0.050, 0.032, 0.032],  # LEAP Thumb
                [0.050, 0.032, 0.032],  # LEAP Middle
                [0.050, 0.032, 0.032],  # LEAP Ring
            ],
            target_link_lengths=[
                [0.054, 0.038, 0.044],  # Allegro Index
                [0.054, 0.038, 0.044],  # Allegro Thumb
                [0.054, 0.038, 0.044],  # Allegro Middle
                [0.054, 0.038, 0.044],  # Allegro Ring
            ]
        )
    elif mapping_type == 'fingertip_ik_3d':
        # Swap link lengths for inverse direction
        return FingertipIK3DMapping(
            leap_lower, leap_upper,
            allegro_lower, allegro_upper,
            LEAP_TO_ALLEGRO_INDICES,
            device,
            source_link_lengths=[
                [0.050, 0.032, 0.032],  # LEAP Index
                [0.050, 0.032, 0.032],  # LEAP Thumb
                [0.050, 0.032, 0.032],  # LEAP Middle
                [0.050, 0.032, 0.032],  # LEAP Ring
            ],
            target_link_lengths=[
                [0.054, 0.038, 0.044],  # Allegro Index
                [0.054, 0.038, 0.044],  # Allegro Thumb
                [0.054, 0.038, 0.044],  # Allegro Middle
                [0.054, 0.038, 0.044],  # Allegro Ring
            ]
        )
    elif mapping_type == 'calibrated':
        if calibration_file is None:
            calibration_file = 'cache/leap_to_allegro_calibration.npz'
        return CalibratedPolynomialMapping(
            leap_lower, leap_upper,
            allegro_lower, allegro_upper,
            LEAP_TO_ALLEGRO_INDICES,
            device,
            calibration_file=calibration_file
        )
    else:
        raise ValueError(f"Unknown mapping type: {mapping_type}")
