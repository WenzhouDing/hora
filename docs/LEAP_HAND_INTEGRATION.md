# LEAP Hand Integration for Hora

This document describes the integration of LEAP Hand into the Hora framework, providing two modes:
1. **AllegroHandHoraWithLeap**: Allegro hand is primary, LEAP mirrors its motion
2. **LeapHandHora**: LEAP hand is primary, runs the trained Allegro policy directly on LEAP

## Overview

### Mode 1: AllegroHandHoraWithLeap (Allegro Primary, LEAP Mirrors)

The task `AllegroHandHoraWithLeap`:
- Loads both Allegro and LEAP hands in the same IsaacGym environment
- Each hand has its own object to manipulate
- Runs the trained Allegro policy on the Allegro hand
- Maps Allegro joint commands to LEAP hand with appropriate scaling
- Each hand uses its own controller (Allegro: torque control, LEAP: position control)

### Mode 2: LeapHandHora (LEAP Primary, Virtual Allegro for Visualization)

The task `LeapHandHora`:
- LEAP hand is the PRIMARY hand that actually manipulates the object
- Virtual Allegro hand shows what the policy "sees" and "outputs"
- Uses bidirectional joint mapping:
  - LEAP joint state → Virtual Allegro space → Policy observation
  - Policy outputs Allegro-style actions → Map to LEAP targets
- Both hands use position control
- Swappable joint mapping architecture for future experimentation

## Files Added/Modified

### New Files

| File | Description |
|------|-------------|
| `hora/tasks/allegro_hand_hora_with_leap.py` | Allegro primary, LEAP mirrors |
| `hora/tasks/leap_hand_hora.py` | LEAP primary, Virtual Allegro for visualization |
| `hora/tasks/leap_hand_grasp.py` | LEAP grasp generation task |
| `hora/utils/joint_mapping.py` | Swappable joint mapping module |
| `configs/task/AllegroHandHoraWithLeap.yaml` | Config for Allegro primary mode |
| `configs/task/LeapHandHora.yaml` | Config for LEAP primary mode |
| `configs/task/LeapHandGrasp.yaml` | Config for LEAP grasp generation |
| `configs/train/AllegroHandHoraWithLeap.yaml` | Training config |
| `configs/train/LeapHandHora.yaml` | Training config |
| `scripts/vis_dual_hand.sh` | Visualization script (Allegro primary) |
| `scripts/vis_leap_hora.sh` | Visualization script (LEAP primary) |
| `scripts/gen_leap_grasp.sh` | LEAP grasp cache generation script |
| `scripts/test_finger_mapping.py` | Finger-by-finger mapping verification test |

### Modified Files

| File | Change |
|------|--------|
| `hora/tasks/__init__.py` | Added `AllegroHandHoraWithLeap`, `LeapHandHora`, and `LeapHandGrasp` to task registry |

## Architecture

### Joint Structure Comparison

Both hands have 16 DOFs, but with different kinematic structures:

**Allegro Hand (DOF ordering from tree traversal):**
- DOFs 0-3: Index finger (joints 0-3: spread, MCP, PIP, DIP)
- DOFs 4-7: Thumb (joints 12-15)
- DOFs 8-11: Middle finger (joints 4-7: spread, MCP, PIP, DIP)
- DOFs 12-15: Ring finger (joints 8-11: spread, MCP, PIP, DIP)

**LEAP Hand (DOF ordering from tree traversal):**
- DOFs 0-3: Index finger (joints "1","0","2","3" - note MCP before spread)
- DOFs 4-7: Thumb (joints "12","13","14","15")
- DOFs 8-11: Middle finger (joints "5","4","6","7" - MCP before spread)
- DOFs 12-15: Ring finger (joints "9","8","10","11" - MCP before spread)

### Joint Mapping

The mapping accounts for kinematic tree differences within each finger:

```python
allegro_to_leap_dof_mapping = [1, 0, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]
```

This maps:
- **Index (DOFs 0-3):** Swap spread/MCP (0↔1), keep PIP/DIP
- **Thumb (DOFs 4-7):** No swap needed (same order in both hands)
- **Middle (DOFs 8-11):** Swap spread/MCP (8↔9), keep PIP/DIP
- **Ring (DOFs 12-15):** Swap spread/MCP (12↔13), keep PIP/DIP

### Joint Scaling

Since joint limits differ between the two hands, values are normalized and scaled:

```python
# Normalize Allegro to [0, 1]
allegro_normalized = (allegro_targets - allegro_lower) / (allegro_upper - allegro_lower)

# Reorder joints
leap_normalized = allegro_normalized[:, allegro_to_leap_mapping]

# Scale to LEAP range
leap_targets = leap_normalized * (leap_upper - leap_lower) + leap_lower
```

### Controller Separation

| Hand | Control Mode | PD Gains | Notes |
|------|--------------|----------|-------|
| Allegro | Torque control | pgain=3, dgain=0.1 | Custom PD loop in code |
| LEAP | Position control | pgain=3, dgain=0.1 | IsaacGym built-in PD |

The Allegro hand maintains its original torque control behavior (as used in Hora), while LEAP uses position control with the same PD gains. This ensures the Allegro policy works correctly while LEAP tracks the mapped joint targets.

### Hand Orientation

The LEAP hand requires rotation to match Allegro's palm orientation:

```python
# Apply rotations: 180° around X, then -90° around Z
rot_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi)
rot_z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)
leap_hand_start_pose.r = rot_z * rot_x  # Apply X first, then Z
```

### World Frame Axis Mapping (Palm Facing Up)

After the rotation (180° X, -90° Z), the world frame axes map to LEAP hand directions:

| Axis | Direction (from palm's perspective) |
|------|-------------------------------------|
| **+X** | Toward ring/pinky side (away from thumb) |
| **-X** | Toward index/thumb side |
| **+Y** | Toward fingertips (forward) |
| **-Y** | Toward wrist (backward) |
| **+Z** | Up (away from palm) |
| **-Z** | Down (into palm) |

**Object Position Tuning Guide:**
- Ball too close to index finger → increase X
- Ball too close to ring finger → decrease X
- Ball too far from fingertips → decrease Y
- Ball too close to fingertips → increase Y
- Ball too high above palm → decrease Z
- Ball too low in palm → increase Z

Current tuned spawn position for grasp generation: `(0.05, -0.03, 0.63)`

## Usage

### Visualization (Allegro Primary - LEAP Mirrors)

To visualize a trained Allegro policy with LEAP hand mirroring:

```bash
./scripts/vis_dual_hand.sh <checkpoint_folder>

# Example:
./scripts/vis_dual_hand.sh hora_v0.0.2
```

### Visualization (LEAP Primary - Run Policy on LEAP)

To run a trained Allegro Hora policy directly on LEAP hand:

```bash
./scripts/vis_leap_hora.sh <checkpoint_folder>

# Example:
./scripts/vis_leap_hora.sh hora_v0.0.2
```

In this mode:
- LEAP hand (right) is the PRIMARY hand manipulating the object
- Virtual Allegro hand (left) shows what the policy sees/outputs

Both scripts expect a checkpoint at:
```
outputs/AllegroHandHora/<checkpoint_folder>/stage1_nn/best.pth
```

### Configuration Options

**AllegroHandHoraWithLeap** (`configs/task/AllegroHandHoraWithLeap.yaml`):

```yaml
env:
  # LEAP hand positioning
  leapHandOffsetX: 0.3      # X offset from Allegro (meters)

  # LEAP hand asset
  leapHandAsset: 'assets/leap_hand/robot.urdf'

  # LEAP PD gains
  leapPgain: 3.0
  leapDgain: 0.1
```

**LeapHandHora** (`configs/task/LeapHandHora.yaml`):

```yaml
env:
  # LEAP hand asset
  leapHandAsset: 'assets/leap_hand/robot.urdf'
  leapPgain: 3.0
  leapDgain: 0.1

  # Virtual Allegro positioning (left of LEAP)
  virtualAllegroOffsetX: -0.3

  # Joint mapping type ('normalized_scale' or 'identity')
  jointMappingType: 'normalized_scale'
```

### Running with Different Parameters

```bash
python train.py task=AllegroHandHoraWithLeap \
    headless=False \
    pipeline=gpu \
    task.env.numEnvs=1 \
    test=True \
    task.env.leapHandOffsetX=0.4 \
    checkpoint=<path_to_checkpoint>
```

### Testing Joint Mapping

To verify the joint mapping is correct, use the finger-by-finger test script:

```bash
python scripts/test_finger_mapping.py
```

This script:
- Moves each finger one joint at a time (Spread → MCP → PIP → DIP)
- Cycles through Index → Thumb → Middle → Ring
- Prints which Allegro DOF maps to which LEAP DOF
- Both hands should move corresponding fingers together

### Generating LEAP Grasp Cache

To generate a grasp cache specifically for LEAP hand orientation (required for `LeapHandHora` when using LEAP-native grasps):

```bash
./scripts/gen_leap_grasp.sh <GPU_ID> <SCALE>

# Example: Generate grasps at scale 0.8
./scripts/gen_leap_grasp.sh 0 0.8
```

**Important notes:**
- Requires `pipeline=cpu` for contact detection
- Uses 20,000 parallel environments for fast collection
- Outputs to: `cache/leap_leap_internal_grasp_50k_s<SCALE>.npy`
- Each grasp contains 23 values: 16 LEAP joint positions + 7 object pose (x,y,z,qx,qy,qz,qw)

To generate grasps for multiple scales (for scale randomization during training):

```bash
# Generate grasps for scales 0.7, 0.75, 0.8, 0.85
for scale in 0.7 0.75 0.8 0.85; do
    ./scripts/gen_leap_grasp.sh 0 $scale
done
```

After generating the cache, update `LeapHandHora.yaml` to use it:

```yaml
env:
  grasp_cache_name: 'leap_leap_internal'  # Uses LEAP-specific grasp cache
```

## Implementation Details

### Class Hierarchy

```
VecTask (base)
    └── AllegroHandHora
            ├── AllegroHandGrasp           (Allegro grasp generation)
            ├── AllegroHandHoraWithLeap    (Allegro primary, LEAP mirrors)
            ├── LeapHandHora               (LEAP primary, Virtual Allegro)
            └── LeapHandGrasp              (LEAP grasp generation)
```

### Joint Mapping Module

The `hora/utils/joint_mapping.py` module provides swappable joint mapping strategies:

```python
from hora.utils.joint_mapping import (
    create_allegro_to_leap_mapping,
    create_leap_to_allegro_mapping,
    NormalizedScaleMapping,  # Maps via [0,1] normalization
    IdentityMapping,         # Direct reordering without scaling
)

# Create mappings
allegro_to_leap = create_allegro_to_leap_mapping(
    allegro_lower, allegro_upper,
    leap_lower, leap_upper,
    mapping_type='normalized_scale',  # or 'identity'
    device='cuda:0'
)

# Use mappings
leap_targets = allegro_to_leap.source_to_target(allegro_actions)
allegro_obs = allegro_to_leap.target_to_source(leap_state)
```

This architecture allows easy addition of new mapping strategies (learned, optimization-based, etc.).

### Key Method Overrides

| Method | Purpose |
|--------|---------|
| `_create_object_asset()` | Load LEAP hand URDF in addition to Allegro |
| `_create_envs()` | Create both hand actors in each environment |
| `pre_physics_step()` | Map Allegro targets to LEAP before simulation |
| `update_low_level_control()` | Apply torques to Allegro, position targets to LEAP |
| `reset_idx()` | Reset both hands to mapped initial poses |
| `compute_observations()` | Use only Allegro DOFs for observations |
| `compute_reward()` | Use only Allegro DOFs for reward calculation |

### DOF State Layout

When both hands are loaded, the DOF state tensor has 32 DOFs per environment:
- DOFs 0-15: Allegro hand
- DOFs 16-31: LEAP hand

The `cur_targets` and `prev_targets` tensors follow the same layout.

## Limitations

1. **Object Synchronization**: Both hands have their own objects. The LEAP object position is reset to mirror the Allegro object (offset by `leapHandOffsetX`), but physics interactions are independent.

2. **Joint Mapping**: The current mapping assumes functional correspondence between same-indexed finger joints. May need refinement for perfect mirroring.

3. **Hand Orientation**: LEAP hand is rotated 180° around X-axis then -90° around Z-axis to match Allegro's palm orientation.

## Future Improvements

- ✅ ~~Add configurable joint mapping for different correspondence strategies~~ (Done: `hora/utils/joint_mapping.py`)
- ✅ ~~Bi-directional mapping (LEAP policy controlling Allegro)~~ (Done: `LeapHandHora`)
- Support for different hand offsets (Y, Z directions)
- Synchronized object physics between hands
- Learned joint mapping strategies (IK-based, neural network, etc.)
- Real-time visualization of mapping quality metrics

## Troubleshooting

### Common Issues

**"shape mismatch" errors**: Usually indicates DOF tensor size mismatches. Ensure all tensor operations account for 32 total DOFs.

**LEAP hand not moving**: Check that `cur_targets[:, 16:]` is being properly updated in `pre_physics_step`.

**Visualization crashes**: Ensure the checkpoint was trained with `AllegroHandHora` (not `AllegroHandHoraWithLeap`).

## References

- [Hora Paper](https://arxiv.org/abs/2210.04887) - In-Hand Object Rotation via Rapid Motor Adaptation
- [LEAP Hand Paper](https://arxiv.org/abs/2309.06440) - Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
- Allegro Hand URDF: `assets/allegro/allegro_internal.urdf`
- LEAP Hand URDF: `assets/leap_hand/robot.urdf`
