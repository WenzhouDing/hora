# LEAP Hand Retargeting Pipeline for Hora RMA

## Overview

This document describes the integration of the LEAP hand into the Hora (Hand-Object Rotation using Rapid motor Adaptation) framework. The goal is to transfer a policy trained on the Allegro hand to the LEAP hand without retraining, using joint retargeting.

## Objective

**Zero-shot transfer** of an in-hand object rotation policy from Allegro hand to LEAP hand:
- Policy was trained entirely on Allegro hand in simulation
- At deployment, retargeting maps between joint spaces
- Adaptation module handles remaining sim-to-real/hand-to-hand gaps

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEAP HAND RETARGETING PIPELINE                            │
│                         (LeapHandHora Task)                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────────────────┐
                        │   Pre-trained Allegro Policy      │
                        │     (PPO + Privileged Info)       │
                        │  outputs/AllegroHandHora/...      │
                        └───────────────┬──────────────────┘
                                        │
                                        ▼
                        ┌──────────────────────────────────┐
                        │    Allegro Actions (16 DOFs)      │
                        │     cur_targets (joint angles)    │
                        └───────────────┬──────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
            ▼                           ▼                           │
    ┌───────────────────┐   ┌──────────────────────────┐           │
    │  Virtual Allegro  │   │   ALLEGRO → LEAP         │           │
    │  (Visualization)  │   │   RETARGETING            │           │
    │                   │   │                          │           │
    │  - Offset -0.3m   │   │  allegro_to_leap_map()   │           │
    │  - Shows policy   │   │                          │           │
    │  - No physics     │   │  Options:                │           │
    └───────────────────┘   │  - normalized_scale      │           │
                            │  - fingertip_ik          │           │
                            │  - calibrated            │           │
                            └────────────┬─────────────┘           │
                                         │                         │
                                         ▼                         │
                            ┌──────────────────────────┐           │
                            │   LEAP Actions (16 DOFs)  │           │
                            │    leap_cur_targets       │           │
                            └────────────┬─────────────┘           │
                                         │                         │
                                         ▼                         │
                            ┌──────────────────────────┐           │
                            │   LEAP Hand (Primary)     │           │
                            │                          │           │
                            │  - PD Position Control   │           │
                            │  - Sphere fingertip      │           │
                            │    collision (r=0.01235) │           │
                            │  - Full physics sim      │           │
                            └────────────┬─────────────┘           │
                                         │                         │
                                         ▼                         │
                            ┌──────────────────────────┐           │
                            │   LEAP Hand State         │           │
                            │  - leap_hand_dof_pos     │           │
                            │  - leap_hand_dof_vel     │           │
                            │  - object_pos/vel        │           │
                            └────────────┬─────────────┘           │
                                         │                         │
                                         ▼                         │
                            ┌──────────────────────────┐           │
                            │   LEAP → ALLEGRO         │           │
                            │   RETARGETING            │           │
                            │                          │           │
                            │  leap_to_allegro_map()   │           │
                            └────────────┬─────────────┘           │
                                         │                         │
                                         ▼                         │
                            ┌──────────────────────────┐           │
                            │  Allegro-Space State      │◄──────────┘
                            │  allegro_hand_dof_pos    │  (sync virtual)
                            │  allegro_hand_dof_vel    │
                            └────────────┬─────────────┘
                                         │
                                         ▼
                            ┌──────────────────────────┐
                            │   Observation Buffer      │
                            │                          │
                            │  - fingertip_pos (4*3)   │
                            │  - dof_pos (16)          │
                            │  - dof_vel (16)          │
                            │  - object_pose (7)       │
                            │  - prev_actions (16)     │
                            │  - proprio_hist (30*24)  │
                            └────────────┬─────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                        │
                    ▼                                        ▼
            ┌───────────────┐                    ┌─────────────────────┐
            │  BASE POLICY  │                    │  ADAPTATION MODULE  │
            │   (Actor)     │                    │  (Priv Encoder)     │
            │               │                    │                     │
            │  obs → action │                    │  proprio_hist →     │
            │               │                    │  priv_info_estimate │
            └───────────────┘                    └─────────────────────┘
```

## Key Insight: Both Policy Components Receive Mapped Data

The LEAP → Allegro mapped state is fed to **BOTH**:
1. **Base Policy**: Receives `obs_buf` containing mapped joint positions/velocities
2. **Adaptation Module**: Receives `proprio_hist_buf` containing history of mapped proprioception

This ensures the pre-trained Allegro policy sees observations in the joint space it was trained on.

---

## Retargeting Methods

### 1. Normalized Scale Mapping (`normalized_scale`)

**Concept**: Map joint positions based on their normalized position within joint limits.

```python
# Normalize source to [0, 1]
normalized = (source_joint - source_lower) / (source_upper - source_lower)

# Denormalize to target range
target_joint = normalized * (target_upper - target_lower) + target_lower
```

**Pros**:
- Simple and fast
- No calibration needed
- Works reasonably for similar hand kinematics

**Cons**:
- Doesn't account for kinematic differences
- Fingertip positions may not match

**Code**: `hora/utils/joint_mapping.py:NormalizedScaleMapping`

---

### 2. Fingertip IK Mapping (`fingertip_ik`)

**Concept**: Match fingertip positions using analytical inverse kinematics in the finger plane.

```
Source Hand              Target Hand
    │                        │
    ▼                        ▼
┌─────────┐            ┌─────────┐
│   FK    │            │   IK    │
│         │            │         │
│ joints  │──►tips────►│  tips   │
│  → pos  │            │ → joints│
└─────────┘            └─────────┘
```

**2D Forward Kinematics** (position relative to MCP joint origin):
```
x = l1·cos(θ1) + l2·cos(θ1+θ2) + l3·cos(θ1+θ2+θ3)
y = l1·sin(θ1) + l2·sin(θ2+θ2) + l3·sin(θ1+θ2+θ3)

Where:
  l1, l2, l3 = link lengths (proximal, middle, distal)
  θ1 = MCP joint angle
  θ2 = PIP joint angle
  θ3 = DIP joint angle
```

**Link Lengths**:
- Allegro: [0.054, 0.038, 0.044]m = 136mm total reach
- LEAP:    [0.050, 0.032, 0.032]m = 114mm total reach

**Algorithm**:
1. Compute fingertip position from source joints using FK (relative to MCP)
2. Use fingertip position directly (no scaling)
3. Solve for target joints using Jacobian transpose method

**Pros**:
- Better fingertip position matching
- Fast analytical solution

**Cons**:
- Simplified 2D (ignores spread joint in FK)
- Spread angle copied directly

**Code**: `hora/utils/joint_mapping.py:FingertipIKMapping`

---

### 3. Full 3D IK Mapping (`fingertip_ik_3d`)

**Concept**: Full 3D inverse kinematics that properly accounts for the spread joint rotation.

**3D Forward Kinematics** (position relative to MCP joint origin):
```
# First compute position in finger plane
x_plane = l1·cos(θ1) + l2·cos(θ1+θ2) + l3·cos(θ1+θ2+θ3)
y_plane = l1·sin(θ1) + l2·sin(θ2+θ2) + l3·sin(θ1+θ2+θ3)

# Apply spread rotation (around Y axis)
x = x_plane · cos(spread)
y = y_plane
z = x_plane · sin(spread)
```

**Coordinate System** (per finger):
- X: along finger when extended (distal direction)
- Y: perpendicular to palm (flexion direction)
- Z: lateral (spread/abduction direction)

**IK Solver**: Damped Least Squares (Levenberg-Marquardt)
```
δq = (JᵀJ + λI)⁻¹ Jᵀ · error

Where:
  J = 3×4 Jacobian matrix (∂[x,y,z]/∂[spread,mcp,pip,dip])
  λ = damping factor (0.05 for stability)
  error = target_pos - current_pos
```

**Optimizations for Performance**:
- Reduced iterations: 5 (down from 20)
- Early termination when error < 1mm
- Step size limiting: ±0.5 rad per iteration
- Pre-allocated damping matrix

**Pros**:
- Full 3D accuracy including spread joint
- Handles out-of-plane fingertip positions

**Cons**:
- Slower than 2D analytical method
- Iterative solver may not converge for extreme positions

**Code**: `hora/utils/joint_mapping.py:FingertipIK3DMapping`

---

### 4. Calibrated Polynomial Mapping (`calibrated`)

**Concept**: Learn a polynomial mapping offline using paired samples.

```
                    OFFLINE CALIBRATION
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Sample Allegro joints → Normalized scale → LEAP   │
│                                                     │
│  Fit polynomial: LEAP = f(Allegro)                 │
│  Fit inverse:    Allegro = g(LEAP)                 │
│                                                     │
│  Features: [1, x₁, x₂, ..., x₁x₁, x₁x₂, ...]      │
│            (153 features for degree 2)             │
│                                                     │
│  Ridge regression: W = (XᵀX + λI)⁻¹XᵀY            │
│                                                     │
└─────────────────────────────────────────────────────┘

                    RUNTIME
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. Normalize input joints to [-1, 1]              │
│  2. Compute polynomial features                     │
│  3. Matrix multiply: output = features @ W          │
│  4. Denormalize to joint limits                    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Polynomial Features** (degree 2):
- Bias: 1
- Linear: x₁, x₂, ..., x₁₆
- Quadratic: x₁², x₁x₂, ..., x₁₆²
- Total: 1 + 16 + 136 = 153 features

**Pros**:
- Can capture non-linear relationships
- Fast at runtime (single matrix multiply)
- Bidirectional (forward and inverse)

**Cons**:
- Requires offline calibration
- Quality depends on ground truth pairing

**Calibration Script**: `scripts/calibrate_hand_mapping.py`

**Code**: `hora/utils/joint_mapping.py:CalibratedPolynomialMapping`

---

## Joint Ordering

### Allegro Hand (16 DOFs)
```
DOFs 0-3:   Index  [spread, MCP, PIP, DIP]
DOFs 4-7:   Thumb  [spread, MCP, PIP, DIP]
DOFs 8-11:  Middle [spread, MCP, PIP, DIP]
DOFs 12-15: Ring   [spread, MCP, PIP, DIP]
```

### LEAP Hand (16 DOFs)
```
DOFs 0-3:   Index  [MCP, spread, PIP, DIP]
DOFs 4-7:   Thumb  [MCP, spread, PIP, DIP]
DOFs 8-11:  Middle [MCP, spread, PIP, DIP]
DOFs 12-15: Ring   [MCP, spread, PIP, DIP]
```

### Index Mapping
```python
# Allegro index → LEAP index
ALLEGRO_TO_LEAP_INDICES = [1, 0, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]

# LEAP index → Allegro index
LEAP_TO_ALLEGRO_INDICES = [1, 0, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]
```

---

## LEAP Hand Physics Setup

### Fingertip Collision
- **Geometry**: Sphere (radius = 0.01235m)
- **Rigid body indices**: 4 (index), 8 (middle), 12 (ring), 16 (thumb)
- **Visual**: Original mesh (fingertip.stl)

### Hand Orientation
```python
# LEAP hand palm-up orientation
# 180° around X, then -90° around Z
rot_x = Quat.from_axis_angle(Vec3(1, 0, 0), pi)
rot_z = Quat.from_axis_angle(Vec3(0, 0, 1), -pi/2)
hand_pose.r = rot_z * rot_x
```

### Object Spawn Position
```python
# Ball spawn position (tuned for grasp generation)
obj_pose.p = Vec3(0.05, -0.03, 0.63)
```

---

## Grasp Cache System

### Two Types of Grasp Caches

1. **LEAP-Native Cache** (`leap_leap_internal`):
   - Generated with `LeapHandGrasp` task
   - Contains LEAP joint positions directly
   - No mapping needed at reset

2. **Allegro Cache** (legacy):
   - Generated with original Allegro task
   - Contains Allegro joint positions
   - Requires Allegro → LEAP mapping at reset

### Grasp Validity Conditions
```python
# All fingertips within 0.1m of object
cond1 = (finger_dists < 0.1).all(-1)

# At least 2 fingers in contact
cond2 = contact_count >= 2

# Object above height threshold
cond3 = obj_z > reset_z_threshold
```

### Cache File Naming
```
cache/leap_{cache_name}_grasp_50k_s{scale}.npy
# Example: cache/leap_leap_internal_grasp_50k_s08.npy
```

---

## Configuration

### LeapHandHora.yaml Key Settings
```yaml
env:
  # Joint mapping method
  jointMappingType: 'fingertip_ik'  # Options: normalized_scale, fingertip_ik, fingertip_ik_3d, calibrated

  # Grasp cache (LEAP-native)
  grasp_cache_name: 'leap_leap_internal'

  # Virtual Allegro visualization
  virtualAllegroOffsetX: -0.3
  showVirtualObject: False

  # LEAP hand asset and PD control
  leapHandAsset: 'assets/leap_hand/robot.urdf'
  leapPgain: 4.0   # Position gain (increased for responsiveness)
  leapDgain: 0.05  # Derivative gain (reduced to decrease damping feel)
```

---

## Code Locations

| Component | File | Line Numbers |
|-----------|------|--------------|
| LeapHandHora Task | `hora/tasks/leap_hand_hora.py` | - |
| Joint Mapping Classes | `hora/utils/joint_mapping.py` | 1-530 |
| Retargeter Setup | `leap_hand_hora.py` | 113-122 |
| Action Retargeting | `leap_hand_hora.py` | 617-620 |
| State Retargeting | `leap_hand_hora.py` | 651-653 |
| Grasp Generation | `hora/tasks/leap_hand_grasp.py` | - |
| Calibration Script | `scripts/calibrate_hand_mapping.py` | - |

---

## What Has Been Done

### 1. LEAP Hand Integration
- [x] Created `LeapHandHora` task inheriting from `AllegroHandHora`
- [x] Dual-hand environment (LEAP primary, Allegro virtual)
- [x] LEAP hand URDF with sphere fingertip collisions
- [x] Correct hand orientation (palm-up)

### 2. Joint Retargeting System
- [x] `NormalizedScaleMapping` - basic range scaling
- [x] `FingertipIKMapping` - 2D analytical IK-based
- [x] `FingertipIK3DMapping` - full 3D damped least squares IK
- [x] `CalibratedPolynomialMapping` - learned polynomial
- [x] Factory functions for easy mapping creation
- [x] Bidirectional mapping (forward + inverse)

### 3. Grasp Generation
- [x] `LeapHandGrasp` task for LEAP-native grasps
- [x] Grasp validity conditions (distance, contacts, height)
- [x] Cache generation script (`gen_leap_grasp.sh`)

### 4. Pipeline Integration
- [x] Action retargeting (Allegro → LEAP) in `pre_physics_step`
- [x] State retargeting (LEAP → Allegro) in `post_physics_step`
- [x] Both base policy and adaptation module receive mapped data
- [x] Virtual Allegro visualization option

### 5. Configuration & Scripts
- [x] `LeapHandHora.yaml` configuration
- [x] `vis_leap_hora.sh` visualization script
- [x] `calibrate_hand_mapping.py` calibration script
- [x] Hide virtual object option

---

## Current Status

- **Mapping Type**: `fingertip_ik` (default)
- **Grasp Cache**: LEAP-native (`leap_leap_internal`)
- **Visualization**: LEAP hand with original fingertip mesh, optional virtual Allegro

---

## Future Work

1. **Better Calibration**: Use Isaac Gym FK for ground truth fingertip positions
2. **Per-Finger Tuning**: Different mapping parameters per finger
3. **Real Robot Deployment**: Test on physical LEAP hand
4. **Fine-tuning**: Optional policy fine-tuning on LEAP if needed
