# LEAP Hand Integration for Hora

This document describes the integration of LEAP Hand into the Hora framework, allowing side-by-side visualization of Allegro Hand policy execution with LEAP Hand mirroring.

## Overview

The integration adds a new task `AllegroHandHoraWithLeap` that:
- Loads both Allegro and LEAP hands in the same IsaacGym environment
- Runs the trained Allegro policy on the Allegro hand
- Maps Allegro joint commands to LEAP hand with appropriate scaling
- Each hand uses its own controller (Allegro: torque control, LEAP: position control)

## Files Added/Modified

### New Files

| File | Description |
|------|-------------|
| `hora/tasks/allegro_hand_hora_with_leap.py` | Main task class extending AllegroHandHora |
| `configs/task/AllegroHandHoraWithLeap.yaml` | Task configuration |
| `configs/train/AllegroHandHoraWithLeap.yaml` | Training configuration |
| `scripts/vis_dual_hand.sh` | Visualization script |
| `scripts/test_finger_mapping.py` | Finger-by-finger mapping verification test |

### Modified Files

| File | Change |
|------|--------|
| `hora/tasks/__init__.py` | Added `AllegroHandHoraWithLeap` to task registry |

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

## Usage

### Visualization

To visualize a trained Allegro policy with LEAP hand mirroring:

```bash
./scripts/vis_dual_hand.sh <checkpoint_folder>

# Example:
./scripts/vis_dual_hand.sh hora_v0.0.2
```

This expects a checkpoint at:
```
outputs/AllegroHandHora/<checkpoint_folder>/stage1_nn/best.pth
```

### Configuration Options

In `configs/task/AllegroHandHoraWithLeap.yaml`:

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

## Implementation Details

### Class Hierarchy

```
VecTask (base)
    └── AllegroHandHora
            └── AllegroHandHoraWithLeap
```

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

1. **Object Manipulation**: Only the Allegro hand holds/manipulates objects. LEAP hand is purely for visualization/comparison.

2. **Joint Mapping**: The current mapping assumes functional correspondence between same-indexed finger joints. May need refinement for perfect mirroring.

3. **Hand Orientation**: LEAP hand is rotated 180° around X-axis then -90° around Z-axis to match Allegro's palm orientation.

## Future Improvements

- Add configurable joint mapping for different correspondence strategies
- Support for LEAP hand also manipulating objects
- Bi-directional mapping (LEAP policy controlling Allegro)
- Support for different hand offsets (Y, Z directions)

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
