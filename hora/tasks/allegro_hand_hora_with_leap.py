# --------------------------------------------------------
# Dual Hand Visualization: Allegro Hand + LEAP Hand
# Extension of Hora for side-by-side hand comparison
# --------------------------------------------------------

import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, tensor_clamp, quat_conjugate, quat_mul, unscale

from .allegro_hand_hora import AllegroHandHora, compute_hand_reward, quat_to_axis_angle


class AllegroHandHoraWithLeap(AllegroHandHora):
    """
    Extended task that loads both Allegro and LEAP hands side by side.
    The Allegro hand runs the trained policy and the LEAP hand mirrors
    the joint commands with appropriate mapping and scaling.
    """

    def __init__(self, config, sim_device, graphics_device_id, headless):
        # LEAP hand specific configuration
        self.leap_hand_offset_x = config['env'].get('leapHandOffsetX', 0.3)
        self.leap_hand_asset_file = config['env'].get('leapHandAsset', 'assets/leap_hand/robot.urdf')

        # Joint mapping from Allegro DOF index to LEAP DOF index
        # This accounts for the different kinematic tree ordering between hands.
        #
        # Finger structure (both hands):
        #   DOFs 0-3: Index finger
        #   DOFs 4-7: Thumb
        #   DOFs 8-11: Middle finger
        #   DOFs 12-15: Ring finger
        #
        # Within-finger mapping:
        #   Index/Middle/Ring: Allegro has spread first (joint x.0), LEAP tree traversal
        #                      puts MCP first. So we swap positions 0↔1, 8↔9, 12↔13.
        #   Thumb: Both hands have same joint ordering (joints 12-15), no swap needed.
        #
        # Mapping: [index_swap, thumb_no_swap, middle_swap, ring_swap]
        self.allegro_to_leap_dof_mapping = [1, 0, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]

        # Call parent constructor
        super().__init__(config, sim_device, graphics_device_id, headless)

        # Setup LEAP hand specific tensors after parent init
        self._setup_leap_tensors()

    def _setup_leap_tensors(self):
        """Setup tensors for LEAP hand state and control."""
        # Get LEAP hand DOF state slice
        # The parent's dof_state tensor is (num_envs, total_dofs, 2)
        # where total_dofs = num_allegro_hand_dofs + num_leap_hand_dofs
        # DOFs are ordered: [Allegro DOFs 0-15, LEAP DOFs 0-15]

        leap_start = self.num_allegro_hand_dofs
        leap_end = leap_start + self.num_leap_hand_dofs

        # Create views into the existing dof_state for LEAP hand
        dof_state_reshaped = self.dof_state.view(self.num_envs, -1, 2)
        self.leap_hand_dof_state = dof_state_reshaped[:, leap_start:leap_end]
        self.leap_hand_dof_pos = self.leap_hand_dof_state[..., 0]
        self.leap_hand_dof_vel = self.leap_hand_dof_state[..., 1]

        # Create mapping tensors for efficient joint command transfer
        self.allegro_to_leap_mapping = to_torch(self.allegro_to_leap_dof_mapping,
                                                 dtype=torch.long, device=self.device)

        print(f"[AllegroHandHoraWithLeap] Total DOFs per env: {self.num_dofs}")
        print(f"[AllegroHandHoraWithLeap] Allegro DOFs: 0-{self.num_allegro_hand_dofs-1}")
        print(f"[AllegroHandHoraWithLeap] LEAP DOFs: {leap_start}-{leap_end-1}")

    def _create_object_asset(self):
        """Override to load both Allegro and LEAP hand assets."""
        # Call parent to load Allegro hand and objects
        super()._create_object_asset()

        # Load LEAP hand asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

        leap_asset_options = gymapi.AssetOptions()
        leap_asset_options.flip_visual_attachments = False
        leap_asset_options.fix_base_link = True
        leap_asset_options.collapse_fixed_joints = True
        leap_asset_options.disable_gravity = True
        leap_asset_options.thickness = 0.001
        leap_asset_options.angular_damping = 0.01
        # LEAP uses position control (not torque control)
        leap_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # Enable convex decomposition for better collision
        leap_asset_options.vhacd_enabled = True
        leap_asset_options.vhacd_params.resolution = 300000

        self.leap_hand_asset = self.gym.load_asset(self.sim, asset_root,
                                                    self.leap_hand_asset_file,
                                                    leap_asset_options)

        # Get LEAP hand DOF count and properties
        self.num_leap_hand_dofs = self.gym.get_asset_dof_count(self.leap_hand_asset)
        assert self.num_leap_hand_dofs == 16, f"Expected 16 LEAP DOFs, got {self.num_leap_hand_dofs}"

        # Get LEAP hand joint limits
        leap_dof_props = self.gym.get_asset_dof_properties(self.leap_hand_asset)
        self.leap_hand_dof_lower_limits = []
        self.leap_hand_dof_upper_limits = []
        for i in range(self.num_leap_hand_dofs):
            self.leap_hand_dof_lower_limits.append(leap_dof_props['lower'][i])
            self.leap_hand_dof_upper_limits.append(leap_dof_props['upper'][i])

        self.leap_hand_dof_lower_limits = to_torch(self.leap_hand_dof_lower_limits, device=self.device)
        self.leap_hand_dof_upper_limits = to_torch(self.leap_hand_dof_upper_limits, device=self.device)

        print(f"[AllegroHandHoraWithLeap] Loaded LEAP hand with {self.num_leap_hand_dofs} DOFs")

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Override to create environments with both hands."""
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()

        # Setup Allegro hand DOF properties
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            allegro_hand_dof_props['effort'][i] = 0.5
            if self.torque_control:
                allegro_hand_dof_props['stiffness'][i] = 0.
                allegro_hand_dof_props['damping'][i] = 0.
                allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                allegro_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                allegro_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            allegro_hand_dof_props['friction'][i] = 0.01
            allegro_hand_dof_props['armature'][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)

        # Setup LEAP hand DOF properties (position control)
        leap_hand_dof_props = self.gym.get_asset_dof_properties(self.leap_hand_asset)
        leap_pgain = self.config['env'].get('leapPgain', 3.0)
        leap_dgain = self.config['env'].get('leapDgain', 0.1)

        for i in range(self.num_leap_hand_dofs):
            leap_hand_dof_props['effort'][i] = 0.95  # LEAP has higher effort limit
            leap_hand_dof_props['stiffness'][i] = leap_pgain
            leap_hand_dof_props['damping'][i] = leap_dgain
            leap_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            leap_hand_dof_props['friction'][i] = 0.01
            leap_hand_dof_props['armature'][i] = 0.001

        # Get poses for both hands
        allegro_hand_pose, obj_pose = self._init_object_pose()
        leap_hand_pose = self._init_leap_hand_pose()

        # Compute aggregate size (now includes both hands)
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        self.num_leap_hand_bodies = self.gym.get_asset_rigid_body_count(self.leap_hand_asset)
        self.num_leap_hand_shapes = self.gym.get_asset_rigid_shape_count(self.leap_hand_asset)

        max_agg_bodies = self.num_allegro_hand_bodies + self.num_leap_hand_bodies + 2
        max_agg_shapes = self.num_allegro_hand_shapes + self.num_leap_hand_shapes + 2

        self.envs = []
        self.object_init_state = []
        self.hand_indices = []
        self.leap_hand_indices = []
        self.object_indices = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        leap_hand_rb_count = self.gym.get_asset_rigid_body_count(self.leap_hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(allegro_hand_rb_count + leap_hand_rb_count,
                                            allegro_hand_rb_count + leap_hand_rb_count + object_rb_count))

        for i in range(num_envs):
            # Create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # Add Allegro hand (primary hand with policy)
            allegro_actor = self.gym.create_actor(env_ptr, self.hand_asset, allegro_hand_pose,
                                                   'allegro_hand', i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_hand_dof_props)
            allegro_idx = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(allegro_idx)

            # Add LEAP hand (mirroring hand)
            leap_actor = self.gym.create_actor(env_ptr, self.leap_hand_asset, leap_hand_pose,
                                                'leap_hand', i, -1, 1)  # Different collision group
            self.gym.set_actor_dof_properties(env_ptr, leap_actor, leap_hand_dof_props)
            leap_idx = self.gym.get_actor_index(env_ptr, leap_actor, gymapi.DOMAIN_SIM)
            self.leap_hand_indices.append(leap_idx)

            # Add object (for Allegro hand to manipulate)
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 0)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # Set object scale and properties (same as parent)
            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(self.randomize_scale_list[i % num_scales] - 0.025,
                                              self.randomize_scale_list[i % num_scales] + 0.025)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                # Set friction for Allegro hand
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, allegro_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, allegro_actor, hand_props)
                # Set friction for object
                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.leap_hand_indices = to_torch(self.leap_hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def _init_leap_hand_pose(self):
        """Initialize LEAP hand pose offset from Allegro hand."""
        leap_hand_start_pose = gymapi.Transform()
        # Offset LEAP hand in X direction
        leap_hand_start_pose.p = gymapi.Vec3(self.leap_hand_offset_x, 0, 0.5)
        # Rotate to match orientation (LEAP needs different rotation)
        # LEAP hand default is palm facing down, we want it facing the same way as Allegro
        # Apply rotations: first 180 deg around X, then -90 deg around Z
        rot_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi)
        rot_z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)
        # Combine rotations: rot_z * rot_x (apply X first, then Z)
        leap_hand_start_pose.r = rot_z * rot_x
        return leap_hand_start_pose

    def map_allegro_to_leap_joints(self, allegro_targets):
        """
        Map Allegro hand joint targets to LEAP hand joint targets.
        This involves:
        1. Reordering joints based on kinematic tree differences
        2. Scaling joint values from Allegro range to LEAP range

        Args:
            allegro_targets: Tensor of shape (num_envs, 16) with Allegro joint targets

        Returns:
            leap_targets: Tensor of shape (num_envs, 16) with LEAP joint targets
        """
        # Get the Allegro joint limits for normalization
        allegro_lower = self.allegro_hand_dof_lower_limits
        allegro_upper = self.allegro_hand_dof_upper_limits

        # Normalize Allegro targets to [0, 1]
        allegro_normalized = (allegro_targets - allegro_lower) / (allegro_upper - allegro_lower + 1e-8)
        allegro_normalized = torch.clamp(allegro_normalized, 0.0, 1.0)

        # Reorder according to joint mapping
        # allegro_to_leap_mapping[i] tells us which LEAP DOF corresponds to Allegro DOF i
        leap_normalized = allegro_normalized[:, self.allegro_to_leap_mapping]

        # Get the LEAP joint limits (need to be in the same order as LEAP DOFs)
        leap_lower = self.leap_hand_dof_lower_limits
        leap_upper = self.leap_hand_dof_upper_limits

        # Scale to LEAP joint range
        leap_targets = leap_normalized * (leap_upper - leap_lower) + leap_lower

        return leap_targets

    def pre_physics_step(self, actions):
        """Override to apply actions to Allegro hand only."""
        # Allegro hand uses the first 16 DOFs
        # The actions are for Allegro hand (16 actions)
        self.actions = actions.clone().to(self.device)

        # Update Allegro targets (only first 16 DOFs)
        allegro_targets = self.prev_targets[:, :self.num_allegro_hand_dofs] + 1 / 24 * self.actions
        self.cur_targets[:, :self.num_allegro_hand_dofs] = tensor_clamp(
            allegro_targets, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits
        )
        self.prev_targets[:, :self.num_allegro_hand_dofs] = self.cur_targets[:, :self.num_allegro_hand_dofs].clone()

        # Map Allegro targets to LEAP targets
        leap_targets = self.map_allegro_to_leap_joints(self.cur_targets[:, :self.num_allegro_hand_dofs])
        self.cur_targets[:, self.num_allegro_hand_dofs:] = leap_targets
        self.prev_targets[:, self.num_allegro_hand_dofs:] = leap_targets

        # Store previous object state
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

        # Apply random forces to object (same as parent)
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def reset_idx(self, env_ids):
        """Override to also reset LEAP hand."""
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch.rand(
                (len(env_ids), self.num_actions), device=self.device
            ) * (self.randomize_p_gain_upper - self.randomize_p_gain_lower) + self.randomize_p_gain_lower
            self.d_gain[env_ids] = torch.rand(
                (len(env_ids), self.num_actions), device=self.device
            ) * (self.randomize_d_gain_upper - self.randomize_d_gain_lower) + self.randomize_d_gain_lower

        # Reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()

            # Set object state
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, 16:]
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0

            # Set Allegro hand state
            allegro_pos = sampled_pose[:, :16]
            self.allegro_hand_dof_pos[s_ids, :] = allegro_pos
            self.allegro_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, :self.num_allegro_hand_dofs] = allegro_pos
            self.cur_targets[s_ids, :self.num_allegro_hand_dofs] = allegro_pos
            # init_pose_buf only stores Allegro poses (first 16 DOFs)
            self.init_pose_buf[s_ids, :self.num_allegro_hand_dofs] = allegro_pos.clone()

            # Set LEAP hand state (mapped from Allegro)
            leap_pos = self.map_allegro_to_leap_joints(allegro_pos)
            self.leap_hand_dof_pos[s_ids, :] = leap_pos
            self.leap_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, self.num_allegro_hand_dofs:] = leap_pos
            self.cur_targets[s_ids, self.num_allegro_hand_dofs:] = leap_pos

        # Apply state changes
        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices), len(object_indices)
        )

        # Set DOF states for both hands
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        leap_indices = self.leap_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.cat([hand_indices, leap_indices])

        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(hand_indices), len(env_ids)
            )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )

        # Reset LEAP DOF states
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets),
            gymtorch.unwrap_tensor(leap_indices), len(env_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(leap_indices), len(env_ids)
        )

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def update_low_level_control(self):
        """Override to update both Allegro and LEAP hand controls."""
        previous_dof_pos = self.allegro_hand_dof_pos.clone()
        self._refresh_gym()

        # Allegro hand control (torque control with PD)
        if self.torque_control:
            dof_pos = self.allegro_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            # dof_vel_finite_diff is only for Allegro hand (16 DOFs)
            # Resize it if needed (first time after parent init)
            if self.dof_vel_finite_diff.shape[1] != self.num_allegro_hand_dofs:
                self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_allegro_hand_dofs),
                                                        device=self.device, dtype=torch.float)
            self.dof_vel_finite_diff = dof_vel.clone()

            # Compute torques for Allegro hand
            allegro_targets = self.cur_targets[:, :self.num_allegro_hand_dofs]
            torques = self.p_gain * (allegro_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torch.clip(torques, -0.5, 0.5).clone()

            # Create full torque tensor (Allegro torques + zeros for LEAP position control)
            full_torques = torch.zeros((self.num_envs, self.num_dofs),
                                       dtype=torch.float, device=self.device)
            full_torques[:, :self.num_allegro_hand_dofs] = self.torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(full_torques))
        else:
            # Position control for Allegro (if not using torque control)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # Set position targets for LEAP hand (always position control)
        # The cur_targets already contains the LEAP targets from pre_physics_step
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def compute_observations(self):
        """Override to use only Allegro DOFs for observations."""
        self._refresh_gym()
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.allegro_hand_dof_pos.shape) * 2.0 - 1.0) * self.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.allegro_hand_dof_pos,
            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        # Only use Allegro targets (first 16 DOFs) for observation
        allegro_targets = self.cur_targets[:, :self.num_allegro_hand_dofs]
        cur_tar_buf = allegro_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
            self.allegro_hand_dof_pos[at_reset_env_ids], self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.allegro_hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        self.obs_buf[:, :t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())

    def compute_reward(self, actions):
        """Override to use only Allegro DOFs for pose diff penalty."""
        self.rot_axis_buf[:, -1] = -1

        # pose diff penalty - only compare Allegro DOFs
        allegro_init_pose = self.init_pose_buf[:, :self.num_allegro_hand_dofs]
        pose_diff_penalty = ((self.allegro_hand_dof_pos - allegro_init_pose) ** 2).sum(-1)

        # work and torque penalty
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2

        # Compute offset in radians. Radians -> radians / sec
        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)

        # linear velocity: use position difference instead of self.object_linvel
        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)

        self.rew_buf[:] = compute_hand_reward(
            object_linvel_penalty, self.object_linvel_penalty_scale,
            rotate_reward, self.rotate_reward_scale,
            pose_diff_penalty, self.pose_diff_penalty_scale,
            torque_penalty, self.torque_penalty_scale,
            work_penalty, self.work_penalty_scale,
        )
        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['object_linvel_penalty'] = object_linvel_penalty.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = object_angvel[:, 0].mean()
        self.extras['pitch'] = object_angvel[:, 1].mean()
        self.extras['yaw'] = object_angvel[:, 2].mean()

    def _refresh_gym(self):
        """Override to update LEAP hand state as well."""
        super()._refresh_gym()
        # LEAP hand state is automatically updated as it's a view into dof_state
