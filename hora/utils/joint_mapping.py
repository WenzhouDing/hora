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

            # Scale target position based on link length ratio
            scale_ratio = self.target_links[finger_idx].sum() / self.source_links[finger_idx].sum()
            target_fingertip = target_fingertip * scale_ratio

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

            # Scale based on link length ratio
            scale_ratio = self.source_links[finger_idx].sum() / self.target_links[finger_idx].sum()
            source_fingertip = source_fingertip * scale_ratio

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
    device: str = 'cuda:0'
) -> JointMappingBase:
    """
    Factory function to create Allegro-to-LEAP joint mapping.

    Args:
        allegro_lower: Allegro hand lower joint limits
        allegro_upper: Allegro hand upper joint limits
        leap_lower: LEAP hand lower joint limits
        leap_upper: LEAP hand upper joint limits
        mapping_type: Type of mapping ('normalized_scale', 'identity', 'fingertip_ik')
        device: Torch device

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
    else:
        raise ValueError(f"Unknown mapping type: {mapping_type}")


def create_leap_to_allegro_mapping(
    leap_lower: torch.Tensor,
    leap_upper: torch.Tensor,
    allegro_lower: torch.Tensor,
    allegro_upper: torch.Tensor,
    mapping_type: str = 'normalized_scale',
    device: str = 'cuda:0'
) -> JointMappingBase:
    """
    Factory function to create LEAP-to-Allegro joint mapping.

    Args:
        leap_lower: LEAP hand lower joint limits
        leap_upper: LEAP hand upper joint limits
        allegro_lower: Allegro hand lower joint limits
        allegro_upper: Allegro hand upper joint limits
        mapping_type: Type of mapping ('normalized_scale', 'identity', 'fingertip_ik')
        device: Torch device

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
    else:
        raise ValueError(f"Unknown mapping type: {mapping_type}")
