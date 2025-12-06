#!/usr/bin/env python3
"""
Test script to verify joint mapping between Allegro and LEAP hands.
Moves one finger at a time on the Allegro hand while LEAP hand mirrors the motion.
"""

import os
import sys
import time
import numpy as np

# Must import isaacgym before torch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch


def main():
    # Initialize gym
    gym = gymapi.acquire_gym()

    # Parse arguments
    args = gymutil.parse_arguments(description="Finger Mapping Test")

    # Simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True

    # Create sim
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("Failed to create sim")
        sys.exit(1)

    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Asset root
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    # Load Allegro hand
    allegro_options = gymapi.AssetOptions()
    allegro_options.fix_base_link = True
    allegro_options.collapse_fixed_joints = True
    allegro_options.disable_gravity = True
    allegro_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    allegro_asset = gym.load_asset(sim, asset_root, 'assets/allegro/allegro.urdf', allegro_options)

    # Load LEAP hand
    leap_options = gymapi.AssetOptions()
    leap_options.fix_base_link = True
    leap_options.collapse_fixed_joints = True
    leap_options.disable_gravity = True
    leap_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    leap_asset = gym.load_asset(sim, asset_root, 'assets/leap_hand/robot.urdf', leap_options)

    # Get DOF properties
    allegro_dof_props = gym.get_asset_dof_properties(allegro_asset)
    leap_dof_props = gym.get_asset_dof_properties(leap_asset)

    # Set PD gains for position control
    for i in range(16):
        allegro_dof_props['stiffness'][i] = 3.0
        allegro_dof_props['damping'][i] = 0.1
        allegro_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        leap_dof_props['stiffness'][i] = 3.0
        leap_dof_props['damping'][i] = 0.1
        leap_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS

    # Joint limits
    allegro_lower = np.array([allegro_dof_props['lower'][i] for i in range(16)])
    allegro_upper = np.array([allegro_dof_props['upper'][i] for i in range(16)])
    leap_lower = np.array([leap_dof_props['lower'][i] for i in range(16)])
    leap_upper = np.array([leap_dof_props['upper'][i] for i in range(16)])

    # Joint mapping (CORRECTED)
    # Index: swap 0↔1, Thumb: no swap, Middle: swap 8↔9, Ring: swap 12↔13
    allegro_to_leap_mapping = [1, 0, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]

    # Create environment
    env_spacing = 1.5
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Allegro hand pose
    allegro_pose = gymapi.Transform()
    allegro_pose.p = gymapi.Vec3(0, -0.2, 0.5)
    allegro_pose.r = gymapi.Quat(0, 0, 0, 1)

    # LEAP hand pose (offset in X, rotated)
    leap_pose = gymapi.Transform()
    leap_pose.p = gymapi.Vec3(0, 0.2, 0.5)
    # Apply rotations: -90 deg around Y
    rot_y = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -np.pi / 2)
    leap_pose.r = rot_y

    # Create actors
    allegro_actor = gym.create_actor(env, allegro_asset, allegro_pose, "allegro", 0, 1)
    leap_actor = gym.create_actor(env, leap_asset, leap_pose, "leap", 0, 1)

    # Set DOF properties
    gym.set_actor_dof_properties(env, allegro_actor, allegro_dof_props)
    gym.set_actor_dof_properties(env, leap_actor, leap_dof_props)

    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        sys.exit(1)

    # Set camera position
    cam_pos = gymapi.Vec3(0.5, -0.5, 0.8)
    cam_target = gymapi.Vec3(0.15, 0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Prepare simulation
    gym.prepare_sim(sim)

    # Get DOF state tensor
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)

    # Finger info
    finger_names = ["Index", "Thumb", "Middle", "Ring"]
    finger_dof_ranges = [(0, 4), (4, 8), (8, 12), (12, 16)]
    joint_names = ["Spread/Base", "MCP", "PIP", "DIP"]

    # Test parameters
    cycle_duration = 10.0  # seconds per finger
    joint_cycle_duration = cycle_duration / 4  # time per joint within finger

    print("\n" + "=" * 60)
    print("  Finger Mapping Test")
    print("=" * 60)
    print("\nThis test will move each finger one at a time.")
    print("Watch both hands - corresponding fingers should move together.\n")
    print("Finger DOF groupings:")
    print("  Index:  Allegro DOFs 0-3  -> LEAP DOFs 1,0,2,3")
    print("  Thumb:  Allegro DOFs 4-7  -> LEAP DOFs 4,5,6,7 (no swap)")
    print("  Middle: Allegro DOFs 8-11 -> LEAP DOFs 9,8,10,11")
    print("  Ring:   Allegro DOFs 12-15 -> LEAP DOFs 13,12,14,15")
    print("\nPress ESC to exit.\n")

    start_time = time.time()
    current_finger = 0
    current_joint = 0

    while not gym.query_viewer_has_closed(viewer):
        # Calculate which finger and joint to move
        elapsed = time.time() - start_time
        # finger_idx = int(elapsed / cycle_duration) % 4
        finger_idx = 0
        joint_in_finger = int((elapsed % cycle_duration) / joint_cycle_duration) % 4

        # Sine wave for smooth motion
        t = (elapsed % joint_cycle_duration) / joint_cycle_duration
        wave = 0.5 * (1 - np.cos(2 * np.pi * t))  # 0 to 1 to 0

        # Print status when finger or joint changes
        if finger_idx != current_finger or joint_in_finger != current_joint:
            current_finger = finger_idx
            current_joint = joint_in_finger
            finger_name = finger_names[finger_idx]
            joint_name = joint_names[joint_in_finger]
            dof_start, dof_end = finger_dof_ranges[finger_idx]
            allegro_dof = dof_start + joint_in_finger
            leap_dof = allegro_to_leap_mapping[allegro_dof]
            print(f"Moving: {finger_name} - {joint_name} (Allegro DOF {allegro_dof} -> LEAP DOF {leap_dof})")

        # Calculate target positions
        allegro_targets = np.zeros(16)
        leap_targets = np.zeros(16)

        # Set neutral position for all joints (middle of range)
        for i in range(16):
            allegro_targets[i] = (allegro_lower[i] + allegro_upper[i]) / 2
            leap_targets[i] = (leap_lower[i] + leap_upper[i]) / 2

        # Move current joint
        dof_start, dof_end = finger_dof_ranges[finger_idx]
        allegro_dof = dof_start + joint_in_finger

        # Interpolate from lower to upper limit
        allegro_targets[allegro_dof] = allegro_lower[allegro_dof] + wave * (
            allegro_upper[allegro_dof] - allegro_lower[allegro_dof]
        )

        # Map to LEAP: normalize and scale
        for i in range(16):
            # Normalize Allegro position to [0, 1]
            allegro_range = allegro_upper[i] - allegro_lower[i]
            if allegro_range > 0:
                normalized = (allegro_targets[i] - allegro_lower[i]) / allegro_range
            else:
                normalized = 0.5
            normalized = np.clip(normalized, 0, 1)

            # Map to LEAP DOF and scale
            leap_dof_idx = allegro_to_leap_mapping[i]
            leap_range = leap_upper[leap_dof_idx] - leap_lower[leap_dof_idx]
            leap_targets[leap_dof_idx] = leap_lower[leap_dof_idx] + normalized * leap_range

        # Set targets (Allegro: DOFs 0-15, LEAP: DOFs 16-31)
        targets = np.concatenate([allegro_targets, leap_targets])
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(
            torch.tensor(targets, dtype=torch.float32, device='cuda:0')
        ))

        # Step simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Update viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Sync to real time
        gym.sync_frame_time(sim)

    # Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    print("\nTest complete.")


if __name__ == "__main__":
    main()
