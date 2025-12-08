# --------------------------------------------------------
# LEAP Hand Hora: Running Allegro policy on LEAP hand
# LEAP is primary, Virtual Allegro for visualization
# --------------------------------------------------------

import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, tensor_clamp, unscale, quat_mul, quat_conjugate

from .allegro_hand_hora import AllegroHandHora, compute_hand_reward, quat_to_axis_angle

from ..utils.joint_mapping import (
    create_allegro_to_leap_mapping,
    create_leap_to_allegro_mapping,
    ALLEGRO_TO_LEAP_INDICES,
    LEAP_TO_ALLEGRO_INDICES
)


class LeapHandHora(AllegroHandHora):
    """
    Run trained Allegro Hora policy on LEAP hand.

    Architecture:
    - LEAP hand is the PRIMARY hand (position control, manipulates object)
    - Virtual Allegro hand is for VISUALIZATION only (shows what policy sees/outputs)

    Data flow:
    1. LEAP joint state -> map to Virtual Allegro space -> Policy observation
    2. Policy outputs Allegro-style actions
    3. Actions -> map to LEAP joint targets -> LEAP position control
    4. Virtual Allegro hand visualizes the mapped state
    """

    def __init__(self, config, sim_device, graphics_device_id, headless):
        # Configuration
        self.virtual_allegro_offset_x = config['env'].get('virtualAllegroOffsetX', -0.3)
        self.leap_hand_asset_file = config['env'].get('leapHandAsset', 'assets/leap_hand/robot.urdf')
        self.mapping_type = config['env'].get('jointMappingType', 'normalized_scale')

        # DOF index mappings (set before parent init)
        self.allegro_to_leap_indices = ALLEGRO_TO_LEAP_INDICES
        self.leap_to_allegro_indices = LEAP_TO_ALLEGRO_INDICES

        # Call parent constructor
        super().__init__(config, sim_device, graphics_device_id, headless)

        # Setup joint mapping after parent init (needs joint limits)
        self._setup_joint_mapping()

        # Setup additional tensors
        self._setup_leap_tensors()

    def _setup_joint_mapping(self):
        """Initialize bidirectional joint mapping modules."""
        # LEAP to Allegro (for converting LEAP state to policy observation)
        self.leap_to_allegro_mapping = create_leap_to_allegro_mapping(
            self.leap_hand_dof_lower_limits,
            self.leap_hand_dof_upper_limits,
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits,
            mapping_type=self.mapping_type,
            device=self.device
        )

        # Allegro to LEAP (for converting policy actions to LEAP targets)
        self.allegro_to_leap_mapping = create_allegro_to_leap_mapping(
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits,
            self.leap_hand_dof_lower_limits,
            self.leap_hand_dof_upper_limits,
            mapping_type=self.mapping_type,
            device=self.device
        )

        print(f"[LeapHandHora] Joint mapping type: {self.mapping_type}")

    def _setup_leap_tensors(self):
        """Setup tensors for LEAP hand (primary) state and control."""
        # LEAP hand DOF indices in the full DOF state tensor
        # Layout: [Virtual Allegro DOFs 0-15, LEAP DOFs 16-31]
        leap_start = self.num_allegro_hand_dofs
        leap_end = leap_start + self.num_leap_hand_dofs

        # Create views into the existing dof_state for LEAP hand
        dof_state_reshaped = self.dof_state.view(self.num_envs, -1, 2)
        self.leap_hand_dof_state = dof_state_reshaped[:, leap_start:leap_end]
        self.leap_hand_dof_pos = self.leap_hand_dof_state[..., 0]
        self.leap_hand_dof_vel = self.leap_hand_dof_state[..., 1]

        # Virtual Allegro state (for policy observations)
        # This is computed from LEAP state via mapping
        self.virtual_allegro_pos = torch.zeros(
            (self.num_envs, self.num_allegro_hand_dofs),
            device=self.device, dtype=torch.float
        )
        self.virtual_allegro_targets = torch.zeros(
            (self.num_envs, self.num_allegro_hand_dofs),
            device=self.device, dtype=torch.float
        )

        # LEAP-specific velocity tracking (for work penalty calculation)
        self.leap_dof_vel_finite_diff = torch.zeros(
            (self.num_envs, self.num_leap_hand_dofs),
            device=self.device, dtype=torch.float
        )

        print(f"[LeapHandHora] Total DOFs per env: {self.num_dofs}")
        print(f"[LeapHandHora] Virtual Allegro DOFs: 0-{self.num_allegro_hand_dofs-1}")
        print(f"[LeapHandHora] LEAP DOFs (primary): {leap_start}-{leap_end-1}")

    def _create_object_asset(self):
        """Load both hand assets and objects."""
        # Call parent to load Allegro hand and objects
        super()._create_object_asset()

        # Load LEAP hand asset (this will be the PRIMARY hand)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

        leap_asset_options = gymapi.AssetOptions()
        leap_asset_options.flip_visual_attachments = False
        leap_asset_options.fix_base_link = True
        leap_asset_options.collapse_fixed_joints = True
        leap_asset_options.disable_gravity = True
        leap_asset_options.thickness = 0.001
        leap_asset_options.angular_damping = 0.01
        leap_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        leap_asset_options.vhacd_enabled = True
        leap_asset_options.vhacd_params.resolution = 300000

        self.leap_hand_asset = self.gym.load_asset(
            self.sim, asset_root, self.leap_hand_asset_file, leap_asset_options
        )

        # Get LEAP hand DOF properties
        self.num_leap_hand_dofs = self.gym.get_asset_dof_count(self.leap_hand_asset)
        assert self.num_leap_hand_dofs == 16, f"Expected 16 LEAP DOFs, got {self.num_leap_hand_dofs}"

        # Store LEAP joint limits
        leap_dof_props = self.gym.get_asset_dof_properties(self.leap_hand_asset)
        self.leap_hand_dof_lower_limits = to_torch(
            [leap_dof_props['lower'][i] for i in range(self.num_leap_hand_dofs)],
            device=self.device
        )
        self.leap_hand_dof_upper_limits = to_torch(
            [leap_dof_props['upper'][i] for i in range(self.num_leap_hand_dofs)],
            device=self.device
        )

        print(f"[LeapHandHora] Loaded LEAP hand with {self.num_leap_hand_dofs} DOFs")

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Create environments with LEAP as primary, Virtual Allegro for visualization."""
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Load assets first (Allegro hand, LEAP hand, objects)
        self._create_object_asset()

        # Get number of DOFs for Allegro hand
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)

        # Setup Virtual Allegro hand DOF properties (position control for visualization)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            # Virtual Allegro uses position control (just visualization)
            allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            allegro_hand_dof_props['stiffness'][i] = 3.0
            allegro_hand_dof_props['damping'][i] = 0.1
            allegro_hand_dof_props['friction'][i] = 0.01
            allegro_hand_dof_props['armature'][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(
            self.allegro_hand_dof_lower_limits, device=self.device
        )
        self.allegro_hand_dof_upper_limits = to_torch(
            self.allegro_hand_dof_upper_limits, device=self.device
        )

        # Setup LEAP hand DOF properties (PRIMARY - position control)
        leap_hand_dof_props = self.gym.get_asset_dof_properties(self.leap_hand_asset)
        leap_pgain = self.config['env'].get('leapPgain', 3.0)
        leap_dgain = self.config['env'].get('leapDgain', 0.1)

        for i in range(self.num_leap_hand_dofs):
            leap_hand_dof_props['effort'][i] = 0.95
            leap_hand_dof_props['stiffness'][i] = leap_pgain
            leap_hand_dof_props['damping'][i] = leap_dgain
            leap_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            leap_hand_dof_props['friction'][i] = 0.01
            leap_hand_dof_props['armature'][i] = 0.001

        # Get poses
        leap_hand_pose, leap_obj_pose = self._init_leap_poses()
        virtual_allegro_pose = self._init_virtual_allegro_pose()
        virtual_allegro_obj_pose = self._init_virtual_allegro_object_pose(leap_obj_pose)

        # Compute aggregate size
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        self.num_leap_hand_bodies = self.gym.get_asset_rigid_body_count(self.leap_hand_asset)
        self.num_leap_hand_shapes = self.gym.get_asset_rigid_shape_count(self.leap_hand_asset)

        max_agg_bodies = self.num_allegro_hand_bodies + self.num_leap_hand_bodies + 4
        max_agg_shapes = self.num_allegro_hand_shapes + self.num_leap_hand_shapes + 4

        self.envs = []
        self.object_init_state = []  # LEAP object (primary)
        self.virtual_allegro_object_init_state = []
        self.hand_indices = []  # Virtual Allegro indices
        self.leap_hand_indices = []  # LEAP indices (primary)
        self.object_indices = []  # LEAP object indices (primary)
        self.virtual_allegro_object_indices = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        leap_hand_rb_count = self.gym.get_asset_rigid_body_count(self.leap_hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(
            allegro_hand_rb_count + leap_hand_rb_count,
            allegro_hand_rb_count + leap_hand_rb_count + object_rb_count
        ))

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # Add Virtual Allegro hand (visualization only)
            allegro_actor = self.gym.create_actor(
                env_ptr, self.hand_asset, virtual_allegro_pose,
                'virtual_allegro', i, -1, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_hand_dof_props)
            allegro_idx = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(allegro_idx)

            # Add LEAP hand (PRIMARY - actually manipulates object)
            leap_actor = self.gym.create_actor(
                env_ptr, self.leap_hand_asset, leap_hand_pose,
                'leap_hand', i, -1, 1
            )
            self.gym.set_actor_dof_properties(env_ptr, leap_actor, leap_hand_dof_props)
            leap_idx = self.gym.get_actor_index(env_ptr, leap_actor, gymapi.DOMAIN_SIM)
            self.leap_hand_indices.append(leap_idx)

            # Add LEAP object (PRIMARY - actually manipulated)
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, leap_obj_pose, 'object', i, 0, 0
            )
            self.object_init_state.append([
                leap_obj_pose.p.x, leap_obj_pose.p.y, leap_obj_pose.p.z,
                leap_obj_pose.r.x, leap_obj_pose.r.y, leap_obj_pose.r.z, leap_obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # Add Virtual Allegro object (visualization only)
            virtual_obj_handle = self.gym.create_actor(
                env_ptr, object_asset, virtual_allegro_obj_pose,
                'virtual_allegro_object', i, 0, 2
            )
            self.virtual_allegro_object_init_state.append([
                virtual_allegro_obj_pose.p.x, virtual_allegro_obj_pose.p.y, virtual_allegro_obj_pose.p.z,
                virtual_allegro_obj_pose.r.x, virtual_allegro_obj_pose.r.y, virtual_allegro_obj_pose.r.z,
                virtual_allegro_obj_pose.r.w, 0, 0, 0, 0, 0, 0
            ])
            virtual_obj_idx = self.gym.get_actor_index(env_ptr, virtual_obj_handle, gymapi.DOMAIN_SIM)
            self.virtual_allegro_object_indices.append(virtual_obj_idx)

            # Set object properties (scale, mass, friction, etc.)
            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(
                    self.randomize_scale_list[i % num_scales] - 0.025,
                    self.randomize_scale_list[i % num_scales] + 0.025
                )
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self.gym.set_actor_scale(env_ptr, virtual_obj_handle, obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale)

            # Set friction for LEAP hand and primary object
            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(
                    self.randomize_friction_lower, self.randomize_friction_upper
                )
                leap_hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, leap_actor)
                for p in leap_hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, leap_actor, leap_hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)

            # Set mass
            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            # Set COM
            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                obj_com = [
                    np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                    np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                    np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)
                ]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        # Convert to tensors
        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.virtual_allegro_object_init_state = to_torch(
            self.virtual_allegro_object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.leap_hand_indices = to_torch(self.leap_hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.virtual_allegro_object_indices = to_torch(
            self.virtual_allegro_object_indices, dtype=torch.long, device=self.device
        )

    def _init_leap_poses(self):
        """Initialize LEAP hand and object poses (PRIMARY)."""
        # LEAP hand at origin
        leap_hand_pose = gymapi.Transform()
        leap_hand_pose.p = gymapi.Vec3(0, 0, 0.5)
        # LEAP needs specific rotation to match palm-up orientation
        # 180° around X, then -90° around Z
        rot_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi)
        rot_z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)
        leap_hand_pose.r = rot_z * rot_x

        # Object pose (same as original Allegro object position relative to hand)
        obj_pose = gymapi.Transform()
        obj_z = 0.66 if self.save_init_pose else 0.65
        if 'internal' not in self.grasp_cache_name:
            obj_z -= 0.02
        obj_pose.p = gymapi.Vec3(-0.01, -0.01, obj_z)
        obj_pose.r = gymapi.Quat(0, 0, 0, 1)

        return leap_hand_pose, obj_pose

    def _init_virtual_allegro_pose(self):
        """Initialize Virtual Allegro hand pose (offset from LEAP)."""
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.virtual_allegro_offset_x, 0, 0.5)
        pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), -np.pi / 2
        ) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)
        return pose

    def _init_virtual_allegro_object_pose(self, leap_obj_pose):
        """Initialize Virtual Allegro object pose (offset from LEAP object)."""
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(
            leap_obj_pose.p.x + self.virtual_allegro_offset_x,
            leap_obj_pose.p.y,
            leap_obj_pose.p.z
        )
        pose.r = leap_obj_pose.r
        return pose

    def pre_physics_step(self, actions):
        """
        Process actions: map Allegro-style actions to LEAP targets.

        Flow:
        1. Receive Allegro-style actions from policy
        2. Compute virtual Allegro targets (for visualization)
        3. Map to LEAP joint targets
        4. Apply to LEAP hand via position control
        5. Update virtual Allegro hand visualization
        """
        self.actions = actions.clone().to(self.device)

        # Compute Virtual Allegro targets (what the policy thinks it's commanding)
        virtual_allegro_prev = self.virtual_allegro_targets.clone()
        self.virtual_allegro_targets = virtual_allegro_prev + (1.0 / 24.0) * self.actions
        self.virtual_allegro_targets = tensor_clamp(
            self.virtual_allegro_targets,
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        )

        # Map Virtual Allegro targets to LEAP targets
        leap_targets = self.allegro_to_leap_mapping.source_to_target(self.virtual_allegro_targets)

        # Update target tensors
        # Virtual Allegro (DOFs 0-15): for visualization
        self.cur_targets[:, :self.num_allegro_hand_dofs] = self.virtual_allegro_targets
        self.prev_targets[:, :self.num_allegro_hand_dofs] = self.virtual_allegro_targets.clone()

        # LEAP (DOFs 16-31): actual control
        self.cur_targets[:, self.num_allegro_hand_dofs:] = leap_targets
        self.prev_targets[:, self.num_allegro_hand_dofs:] = leap_targets.clone()

        # Store previous object state
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

        # Apply random forces to LEAP object
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            obj_mass = to_torch([
                self.gym.get_actor_rigid_body_properties(
                    env, self.gym.find_actor_handle(env, 'object')
                )[0].mass for env in self.envs
            ], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device
            ) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE
            )

    def reset_idx(self, env_ids):
        """Reset environments with proper mapping."""
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch.rand(
                (len(env_ids), self.num_actions), device=self.device
            ) * (self.randomize_p_gain_upper - self.randomize_p_gain_lower) + self.randomize_p_gain_lower
            self.d_gain[env_ids] = torch.rand(
                (len(env_ids), self.num_actions), device=self.device
            ) * (self.randomize_d_gain_upper - self.randomize_d_gain_lower) + self.randomize_d_gain_lower

        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue

            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            sampled_pose_idx = np.random.randint(
                self.saved_grasping_states[scale_key].shape[0], size=len(s_ids)
            )
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()

            # Check if using LEAP-native grasp cache (contains 'leap_leap' in name)
            # LEAP-native: sampled_pose[:, :16] = LEAP joint positions
            # Allegro-native: sampled_pose[:, :16] = Allegro joint positions
            use_leap_native = 'leap_leap' in self.grasp_cache_name

            if use_leap_native:
                # LEAP-native cache: positions are already in LEAP space
                leap_pos = sampled_pose[:, :16]
                # Map LEAP to Allegro for virtual hand visualization
                allegro_pos = self.allegro_to_leap_mapping.target_to_source(leap_pos)
            else:
                # Allegro-native cache: map Allegro to LEAP
                allegro_pos = sampled_pose[:, :16]
                leap_pos = self.allegro_to_leap_mapping.source_to_target(allegro_pos)

            # Set Virtual Allegro state
            self.allegro_hand_dof_pos[s_ids, :] = allegro_pos
            self.allegro_hand_dof_vel[s_ids, :] = 0
            self.virtual_allegro_pos[s_ids, :] = allegro_pos.clone()
            self.virtual_allegro_targets[s_ids, :] = allegro_pos.clone()
            self.leap_hand_dof_pos[s_ids, :] = leap_pos
            self.leap_hand_dof_vel[s_ids, :] = 0

            # Update targets
            self.prev_targets[s_ids, :self.num_allegro_hand_dofs] = allegro_pos
            self.cur_targets[s_ids, :self.num_allegro_hand_dofs] = allegro_pos
            self.prev_targets[s_ids, self.num_allegro_hand_dofs:] = leap_pos
            self.cur_targets[s_ids, self.num_allegro_hand_dofs:] = leap_pos

            # Init pose buffer (in Allegro space for policy)
            self.init_pose_buf[s_ids, :self.num_allegro_hand_dofs] = allegro_pos.clone()

            # Get object pose from grasp cache
            obj_pose = sampled_pose[:, 16:].clone()

            # Set LEAP object state (PRIMARY) - use grasp cache position directly
            self.root_state_tensor[self.object_indices[s_ids], :7] = obj_pose
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0

            # Set Virtual Allegro object state (offset from LEAP)
            virtual_obj_state = obj_pose.clone()
            virtual_obj_state[:, 0] += self.virtual_allegro_offset_x
            self.root_state_tensor[self.virtual_allegro_object_indices[s_ids], :7] = virtual_obj_state
            self.root_state_tensor[self.virtual_allegro_object_indices[s_ids], 7:13] = 0

        # Apply state changes for both objects
        leap_obj_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        virtual_obj_indices = torch.unique(self.virtual_allegro_object_indices[env_ids]).to(torch.int32)
        all_object_indices = torch.cat([leap_obj_indices, virtual_obj_indices])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_object_indices), len(all_object_indices)
        )

        # Set DOF states
        allegro_indices = self.hand_indices[env_ids].to(torch.int32)
        leap_indices = self.leap_hand_indices[env_ids].to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets),
            gymtorch.unwrap_tensor(allegro_indices), len(env_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(allegro_indices), len(env_ids)
        )

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
        """
        Apply position control to both hands.

        Both hands use position control:
        - LEAP: primary control (actually manipulates object)
        - Virtual Allegro: visualization only
        """
        previous_leap_pos = self.leap_hand_dof_pos.clone()
        self._refresh_gym()

        # Compute LEAP velocity for work penalty (even though using position control)
        leap_vel = (self.leap_hand_dof_pos - previous_leap_pos) / self.dt
        self.leap_dof_vel_finite_diff = leap_vel.clone()

        # Compute equivalent "torques" for reward calculation
        # This is approximate - using the PD error as a proxy
        leap_targets = self.cur_targets[:, self.num_allegro_hand_dofs:]
        virtual_torques = self.p_gain * (
            self.virtual_allegro_targets - self.virtual_allegro_pos
        ) - self.d_gain * self.allegro_hand_dof_vel
        self.torques = torch.clip(virtual_torques, -0.5, 0.5).clone()

        # Resize dof_vel_finite_diff if needed
        if self.dof_vel_finite_diff.shape[1] != self.num_allegro_hand_dofs:
            self.dof_vel_finite_diff = torch.zeros(
                (self.num_envs, self.num_allegro_hand_dofs),
                device=self.device, dtype=torch.float
            )

        # Map LEAP velocity to Allegro space for the penalty
        # LEAP → Allegro: use source_to_target (source=LEAP, target=Allegro)
        self.dof_vel_finite_diff = self.leap_to_allegro_mapping.source_to_target(
            self.leap_dof_vel_finite_diff
        )

        # Apply position control to both hands
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

    def compute_observations(self):
        """
        Compute observations from LEAP state, mapped to Allegro space.

        The policy expects Allegro-style observations, so we:
        1. Read LEAP joint state
        2. Map to Virtual Allegro space
        3. Create observations in Allegro format
        """
        self._refresh_gym()

        # Map LEAP state to Virtual Allegro space for policy observation
        # LEAP → Allegro: use source_to_target (source=LEAP, target=Allegro)
        self.virtual_allegro_pos = self.leap_to_allegro_mapping.source_to_target(
            self.leap_hand_dof_pos
        )

        # Also update the actual Allegro DOF view to match (for visualization sync)
        self.allegro_hand_dof_pos[:] = self.virtual_allegro_pos

        # Create observations using Virtual Allegro state
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (
            torch.rand(self.virtual_allegro_pos.shape) * 2.0 - 1.0
        ) * self.joint_noise_scale

        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.virtual_allegro_pos,
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)

        cur_tar_buf = self.virtual_allegro_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # Refill initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
            self.virtual_allegro_pos[at_reset_env_ids],
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = (
            self.virtual_allegro_pos[at_reset_env_ids].unsqueeze(1)
        )

        t_buf = self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1).clone()
        self.obs_buf[:, :t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(
            env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone()
        )

    def compute_reward(self, actions):
        """Compute reward based on LEAP object (primary) rotation."""
        self.rot_axis_buf[:, -1] = -1

        # Pose diff penalty (in Virtual Allegro space)
        allegro_init_pose = self.init_pose_buf[:, :self.num_allegro_hand_dofs]
        pose_diff_penalty = ((self.virtual_allegro_pos - allegro_init_pose) ** 2).sum(-1)

        # Torque and work penalty
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2

        # Object rotation reward (from LEAP object)
        angdiff = quat_to_axis_angle(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))
        )
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)

        # Object linear velocity penalty
        object_linvel = (
            (self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)
        ).clone()
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
        """Refresh gym state tensors."""
        super()._refresh_gym()
        # LEAP state is automatically updated as it's a view into dof_state
