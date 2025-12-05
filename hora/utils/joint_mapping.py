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
        mapping_type: Type of mapping ('normalized_scale', 'identity')
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
        mapping_type: Type of mapping ('normalized_scale', 'identity')
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
    else:
        raise ValueError(f"Unknown mapping type: {mapping_type}")
