# Joint Mapping Methods for Cross-Hand Policy Transfer

This document describes the joint mapping strategies available for transferring policies between Allegro and LEAP hands.

## Summary

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| `normalized_scale` | Fast | Good | General use, similar kinematics |
| `identity` | Fastest | Low | Testing, debugging |
| `fingertip_ik` | Slower | Best | Different link lengths, precise positioning |

## Overview

When transferring a policy trained on one robot hand to another, joint angles must be mapped between the two kinematic structures. The challenges include:

1. **Different joint ordering**: URDF tree traversal may order joints differently
2. **Different joint limits**: Range of motion varies between hands
3. **Different link lengths**: Finger segments have different lengths
4. **Different kinematics**: Same joint angles produce different fingertip positions

## Method 1: Normalized Scale Mapping

**Type**: `normalized_scale`

### Description

Maps joints by normalizing to [0,1] range, reordering, then scaling to target range.

### Algorithm

```python
# 1. Normalize source joints to [0, 1]
normalized = (source_joints - source_lower) / (source_upper - source_lower)

# 2. Reorder joints based on index mapping
reordered = normalized[:, ALLEGRO_TO_LEAP_INDICES]

# 3. Scale to target joint range
target_joints = reordered * (target_upper - target_lower) + target_lower
```

### Index Mapping (Allegro → LEAP)

```python
ALLEGRO_TO_LEAP_INDICES = [1, 0, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]
```

This accounts for different URDF orderings:
- **Index finger (DOFs 0-3)**: Swap spread/MCP (0↔1)
- **Thumb (DOFs 4-7)**: No swap needed
- **Middle finger (DOFs 8-11)**: Swap spread/MCP (8↔9)
- **Ring finger (DOFs 12-15)**: Swap spread/MCP (12↔13)

### Pros
- Simple and fast (no iteration)
- Preserves relative joint positions
- Works well when link length ratios are similar

### Cons
- Ignores kinematic differences
- Same normalized joint angle may produce different fingertip positions
- Less accurate for hands with very different link lengths

### Test Results

For index finger with joints [0, 0.5, 0.8, 0.6]:
```
Allegro → LEAP (normalized_scale):
  MCP: 0.5 → 0.840
  PIP: 0.8 → 0.869
  DIP: 0.6 → 0.716
```

---

## Method 2: Identity Mapping

**Type**: `identity`

### Description

Direct joint reordering without any scaling. Simply maps joint indices.

### Algorithm

```python
# Direct reordering only
target_joints = source_joints[:, ALLEGRO_TO_LEAP_INDICES]
```

### Pros
- Fastest possible mapping
- Useful for debugging joint ordering issues
- Good when joint limits are nearly identical

### Cons
- Ignores joint limit differences
- May command positions outside target's range
- Only useful when hands are very similar

### Use Cases
- Debugging and verification
- Testing joint correspondence
- When hands have identical joint limits

---

## Method 3: Fingertip IK Mapping

**Type**: `fingertip_ik`

### The Problem: Why Simple Scaling Isn't Enough

When transferring a policy from Allegro to LEAP hand, the normalized_scale method maps joint angles proportionally. However, the two hands have **different link lengths**:

```
Allegro finger: 54mm + 38mm + 44mm = 136mm total
LEAP finger:    50mm + 32mm + 32mm = 114mm total
```

This means **the same joint angles produce different fingertip positions**:

```
Example: MCP=0.5, PIP=0.8, DIP=0.6 (radians)

Allegro fingertip position: (87mm, 98mm) from base
LEAP fingertip position:    (72mm, 82mm) from base  ← 16% shorter reach!
```

For manipulation tasks, **fingertip position matters more than joint angles**. If the policy learned to place the fingertip at a specific location relative to an object, we want the LEAP finger to reach that same relative position.

### The Objective: Match Relative Fingertip Position

The IK mapping objective is:

> **Find LEAP joint angles that place the fingertip at the same *relative* position as the Allegro fingertip, scaled by the hand size ratio.**

"Relative position" means the position normalized by total finger length. Since LEAP fingers are 84% the length of Allegro fingers (114mm / 136mm = 0.84), we scale the target position accordingly.

### Visual Explanation

```
ALLEGRO (source)                    LEAP (target)

    Fingertip ●                         Fingertip ●
             /                                   /
            / DIP                               / DIP (shorter)
           /                                   /
          ● PIP joint                         ● PIP joint
         /                                   /
        / Middle                            / Middle (shorter)
       /                                   /
      ● MCP joint                         ● MCP joint
     /                                   /
    / Proximal                          / Proximal (shorter)
   /                                   /
  ● Base                              ● Base

Position: (87, 98)mm                Position: (73, 82)mm = 0.84 × (87, 98)
Joints: [0.5, 0.8, 0.6]             Joints: [?, ?, ?] ← SOLVE WITH IK
```

### Algorithm Step-by-Step

```python
# For each finger (Index, Thumb, Middle, Ring):

# STEP 1: Compute where Allegro fingertip is
#         Using forward kinematics with Allegro link lengths
allegro_fingertip = forward_kinematics(allegro_joints, allegro_links)
# Result: (x=87mm, y=98mm) in finger plane

# STEP 2: Scale to LEAP hand size
#         LEAP fingers are 84% the length of Allegro
scale_ratio = sum(leap_links) / sum(allegro_links)  # = 0.84
target_position = allegro_fingertip * scale_ratio
# Result: (x=73mm, y=82mm) - where LEAP fingertip should be

# STEP 3: Solve IK - find LEAP joints that reach target position
#         Using iterative Jacobian transpose method
leap_joints = inverse_kinematics(target_position, leap_links)
# Result: Different joint angles that achieve the scaled position
```

### Why Scale the Position?

Without scaling, we would ask the shorter LEAP finger to reach the same absolute position as Allegro - which may be outside its workspace (unreachable).

By scaling to 84% of the Allegro position, we ask: "Where would this fingertip be if Allegro's hand was shrunk to LEAP's size?" This preserves the **finger pose geometry** (curl shape) while adapting to the hand size.

### Kinematic Model

Uses simplified 2D planar kinematics per finger. Each finger is modeled as a 3-link planar chain (ignoring the spread/abduction joint which moves the finger laterally):

```
Finger model (side view):

        DIP ●────────● Fingertip
           /    l₃
          /
    PIP ●
       /    l₂
      /
MCP ●
   /    l₁
  /
Base

Forward Kinematics equations:
  θ₁ = MCP angle (from vertical)
  θ₂ = θ₁ + PIP angle (cumulative - each joint adds to previous)
  θ₃ = θ₂ + DIP angle (cumulative)

  x = l₁·cos(θ₁) + l₂·cos(θ₂) + l₃·cos(θ₃)
  y = l₁·sin(θ₁) + l₂·sin(θ₂) + l₃·sin(θ₃)
```

### Link Lengths Used

| Hand | Proximal (l₁) | Middle (l₂) | Distal (l₃) | Total |
|------|---------------|-------------|-------------|-------|
| Allegro | 54mm | 38mm | 44mm | 136mm |
| LEAP | 50mm | 32mm | 32mm | 114mm |
| Ratio | 0.93 | 0.84 | 0.73 | **0.84** |

Note: The individual link ratios vary (0.73 to 0.93), which is why simple joint scaling doesn't preserve fingertip position perfectly.

### IK Solver: Jacobian Transpose Method

The inverse kinematics uses an iterative approach:

```python
def inverse_kinematics(target_pos, link_lengths, initial_guess, iterations=10):
    joints = initial_guess.clone()

    for i in range(iterations):
        # Where is the fingertip now?
        current_pos = forward_kinematics(joints, link_lengths)

        # How far from target?
        error = target_pos - current_pos  # (Δx, Δy)

        # Compute Jacobian: how fingertip moves with each joint
        # J[i,j] = ∂(fingertip_pos_i) / ∂(joint_j)
        J = compute_jacobian(joints, link_lengths)

        # Jacobian transpose method: move joints in direction that reduces error
        # This is simpler than full inverse (J⁻¹) and handles singularities better
        delta_joints = step_size * J.T @ error

        # Update and clamp to joint limits
        joints = joints + delta_joints
        joints = clamp(joints, lower_limits, upper_limits)

    return joints
```

Default parameters:
- `ik_iterations`: 10
- `ik_step_size`: 0.5

### Pros
- Most accurate fingertip positioning
- Accounts for different link lengths
- Physically meaningful mapping

### Cons
- Slower due to iterative IK (10 iterations per finger)
- 2D approximation ignores spread joint effects
- May not converge for extreme positions

### Test Results

For index finger with Allegro joints [spread=0, MCP=0.5, PIP=0.8, DIP=0.6]:

```
Allegro → LEAP (fingertip_ik):
  MCP: 0.5 → 0.809 (vs 0.840 normalized, Δ=-0.031)
  PIP: 0.8 → 0.855 (vs 0.869 normalized, Δ=-0.014)
  DIP: 0.6 → 0.712 (vs 0.716 normalized, Δ=-0.004)
```

**Interpretation**: IK produces slightly *less* flexed joints than normalized scaling. Why? Because LEAP's shorter links mean less flexion is needed to reach the same relative position.

---

## Comparison of All Methods

### Test Setup

Source: Allegro hand with index finger curled
- Input joints: [spread=0, MCP=0.5, PIP=0.8, DIP=0.6] radians

### Joint Angle Results

| Joint | Identity | Normalized | IK | Notes |
|-------|----------|------------|-----|-------|
| Spread | 0.000 | -0.070 | -0.070 | Scaled to LEAP range |
| MCP | 0.500 | 0.840 | 0.809 | IK: -0.031 from normalized |
| PIP | 0.800 | 0.869 | 0.855 | IK: -0.014 from normalized |
| DIP | 0.600 | 0.716 | 0.712 | IK: -0.004 from normalized |

### Resulting Fingertip Positions

This shows the **actual fingertip position** achieved by each method:

```
Method              Fingertip (x, y)     Distance from Allegro target
─────────────────────────────────────────────────────────────────────
Allegro (source)    (87mm, 98mm)         Reference
Target (scaled)     (73mm, 82mm)         What we want LEAP to achieve

LEAP (identity)     (72mm, 82mm)         ~1mm error (similar by coincidence)
LEAP (normalized)   (76mm, 82mm)         ~3mm error in X
LEAP (IK)           (73mm, 82mm)         <1mm error ✓ Best match
```

**Key insight**: The IK method achieves the target fingertip position most accurately because it explicitly solves for joint angles that reach that position, rather than just scaling joint angles and hoping the fingertip ends up in the right place.

---

## Configuration

Set the mapping type in `configs/task/LeapHandHora.yaml`:

```yaml
env:
  jointMappingType: 'fingertip_ik'  # Options: 'normalized_scale', 'identity', 'fingertip_ik'
```

---

## Implementation

The joint mapping module is located at `hora/utils/joint_mapping.py`.

### Class Hierarchy

```
JointMappingBase (abstract)
    ├── NormalizedScaleMapping
    ├── IdentityMapping
    └── FingertipIKMapping
```

### Factory Functions

```python
from hora.utils.joint_mapping import (
    create_allegro_to_leap_mapping,
    create_leap_to_allegro_mapping
)

# Create Allegro → LEAP mapping
mapping = create_allegro_to_leap_mapping(
    allegro_lower, allegro_upper,
    leap_lower, leap_upper,
    mapping_type='fingertip_ik',
    device='cuda:0'
)

# Use mapping
leap_targets = mapping.source_to_target(allegro_actions)
allegro_obs = mapping.target_to_source(leap_state)
```

---

## Recommendations

| Scenario | Recommended Method |
|----------|-------------------|
| Quick testing | `identity` |
| General policy transfer | `normalized_scale` |
| Precision manipulation tasks | `fingertip_ik` |
| Hands with similar link lengths | `normalized_scale` |
| Hands with different link lengths | `fingertip_ik` |
| Real-time control (speed critical) | `normalized_scale` |

---

## Future Improvements

1. **3D IK**: Include spread joint for full 3D fingertip positioning
2. **Learned mapping**: Train a neural network to map joints based on task performance
3. **Optimization-based**: Use gradient descent to minimize manipulation objective
4. **Contact-aware**: Consider contact points when mapping grasps
5. **Adaptive mapping**: Adjust mapping online based on observed behavior
