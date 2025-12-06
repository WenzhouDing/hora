# --------------------------------------------------------
# LEAP Hand Grasp Generation
# Generates grasp poses for LEAP hand orientation
# --------------------------------------------------------

import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis, quat_mul, tensor_clamp, to_torch
import os

from hora.tasks.allegro_hand_hora import AllegroHandHora


class LeapHandGrasp(AllegroHandHora):
    """
    Generate grasp poses for LEAP hand.

    This task uses LEAP hand orientation to sample valid grasps that can be
    used with LeapHandHora. The saved grasps include:
    - 16 LEAP joint positions
    - 7 object pose values (x, y, z, qx, qy, qz, qw)

    Total: 23 values per grasp (same format as Allegro grasp cache).
    """

    def __init__(self, config, sim_device, graphics_device_id, headless):
        # Store LEAP hand asset path
        self.leap_hand_asset_file = config['env'].get('leapHandAsset', 'assets/leap_hand/robot.urdf')

        # Call parent constructor
        super().__init__(config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # Storage for valid grasps
        self.saved_grasping_states = torch.zeros((0, 23), dtype=torch.float, device=self.device)

        # LEAP canonical pose (palm-up, fingers slightly curled)
        # This is designed for LEAP's orientation (180° X, -90° Z rotation)
        # The thumb needs to be more extended to reach the ball
        self.canonical_pose = [
            # Index finger (DOFs 0-3): MCP, spread, PIP, DIP
            1.0, 0.0, 0.4, 0.4,
            # Thumb (DOFs 4-7): More extended to reach toward palm center
            0.8, 1.0, 0.6, 0.3,
            # Middle finger (DOFs 8-11): MCP, spread, PIP, DIP
            1.0, 0.0, 0.4, 0.4,
            # Ring finger (DOFs 12-15): MCP, spread, PIP, DIP
            1.0, 0.0, 0.4, 0.4,
        ]

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _create_object_asset(self):
        """Load LEAP hand instead of Allegro hand."""
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

        # Load LEAP hand asset
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

        self.hand_asset = self.gym.load_asset(
            self.sim, asset_root, self.leap_hand_asset_file, leap_asset_options
        )

        # Get DOF count
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        assert self.num_allegro_hand_dofs == 16, f"Expected 16 LEAP DOFs, got {self.num_allegro_hand_dofs}"

        # Store joint limits
        leap_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        self.allegro_hand_dof_lower_limits = to_torch(
            [leap_dof_props['lower'][i] for i in range(self.num_allegro_hand_dofs)],
            device=self.device
        )
        self.allegro_hand_dof_upper_limits = to_torch(
            [leap_dof_props['upper'][i] for i in range(self.num_allegro_hand_dofs)],
            device=self.device
        )

        print(f"[LeapHandGrasp] Loaded LEAP hand with {self.num_allegro_hand_dofs} DOFs")

        # Load object assets (same as parent)
        self._load_object_assets(asset_root)

    def _load_object_assets(self, asset_root):
        """Load object assets."""
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.fix_base_link = False
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params.resolution = 100000

        self.object_asset_list = []
        self.object_type_list = []

        # Load object based on config
        obj_type = self.config['env']['object']['type']
        if obj_type == 'simple_tennis_ball':
            obj_asset = self.gym.create_sphere(
                self.sim, 0.0325, object_asset_options
            )
            self.object_asset_list.append(obj_asset)
            self.object_type_list.append('simple_tennis_ball')
        else:
            # Load from URDF/mesh
            obj_asset = self.gym.load_asset(
                self.sim, asset_root, f'assets/objects/{obj_type}/mobility.urdf',
                object_asset_options
            )
            self.object_asset_list.append(obj_asset)
            self.object_type_list.append(obj_type)

        self.object_type_prob = self.config['env']['object']['sampleProb']

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Create environments with LEAP hand."""
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Load assets
        self._create_object_asset()

        # Setup DOF properties for position control
        hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        for i in range(self.num_allegro_hand_dofs):
            hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            hand_dof_props['stiffness'][i] = 3.0
            hand_dof_props['damping'][i] = 0.1
            hand_dof_props['friction'][i] = 0.01
            hand_dof_props['armature'][i] = 0.001

        # LEAP hand pose with correct orientation
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0, 0, 0.5)
        # LEAP orientation: 180° around X, then -90° around Z
        rot_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi)
        rot_z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)
        hand_pose.r = rot_z * rot_x

        # Object pose (above hand, will fall into palm)
        # Position tuned for middle finger alignment
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0.05, -0.03, 0.63)  # Lifted 1cm
        obj_pose.r = gymapi.Quat(0, 0, 0, 1)

        # Compute aggregate sizes
        num_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        num_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = num_hand_bodies + 4
        max_agg_shapes = num_hand_shapes + 4

        self.envs = []
        self.object_init_state = []
        self.hand_indices = []
        self.object_indices = []

        # Fingertip rigid body handles (for contact checking)
        # LEAP: fingertip=4, fingertip_2=8, fingertip_3=12, thumb_fingertip=16
        self.object_rb_handles = list(range(num_hand_bodies, num_hand_bodies + 1))

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # Add LEAP hand
            hand_actor = self.gym.create_actor(
                env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # Add object
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, obj_pose, 'object', i, 0, 0
            )
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # Set object scale
            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(
                    self.randomize_scale_list[i % num_scales] - 0.025,
                    self.randomize_scale_list[i % num_scales] + 0.025
                )
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale)

            # Set friction
            if self.randomize_friction:
                rand_friction = np.random.uniform(
                    self.randomize_friction_lower, self.randomize_friction_upper
                )
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                self._update_priv_buf(env_id=i, name='obj_friction', value=rand_friction)
            else:
                self._update_priv_buf(env_id=i, name='obj_friction', value=1.0)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        # Convert to tensors
        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids):
        """Reset environments and save valid grasps."""
        if self.randomize_mass:
            lower, upper = self.randomize_mass_lower, self.randomize_mass_upper
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                for p in prop:
                    p.mass = np.random.uniform(lower, upper)
                self.gym.set_actor_rigid_body_properties(env, handle, prop)
                self._update_priv_buf(env_id=env_id, name='obj_mass', value=prop[0].mass, lower=0, upper=0.2)
        else:
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                self._update_priv_buf(env_id=env_id, name='obj_mass', value=prop[0].mass, lower=0, upper=0.2)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper,
                (len(env_ids), self.num_actions), device=self.device
            ).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper,
                (len(env_ids), self.num_actions), device=self.device
            ).squeeze(1)

        # Generate random values for pose variations
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device
        )

        # Reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # Check which environments completed an episode successfully
        success = self.progress_buf[env_ids] == self.max_episode_length

        # Save successful grasps
        all_states = torch.cat([
            self.allegro_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]
        ], dim=1)
        self.saved_grasping_states = torch.cat([
            self.saved_grasping_states, all_states[env_ids][success]
        ])
        print(f'Current LEAP grasp cache size: {self.saved_grasping_states.shape[0]}')

        # Save cache when we have enough grasps (10 for quick testing)
        # Filename uses "50k" suffix for compatibility with parent class loading
        if len(self.saved_grasping_states) >= 20:
            cache_name = f'cache/leap_{self.grasp_cache_name}_grasp_50k_s{str(self.base_obj_scale).replace(".", "")}.npy'
            np.save(cache_name, self.saved_grasping_states[:20].cpu().numpy())
            print(f'Saved LEAP grasp cache to: {cache_name}')
            exit()

        # Reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx]

        # No rotation randomization for grasp generation
        new_object_rot = torch.zeros((len(env_ids), 4), device=self.device)
        new_object_rot[:, -1] = 1  # Identity quaternion
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices), len(object_indices)
        )

        # Reset hand to canonical pose with variation
        pos = to_torch(self.canonical_pose, device=self.device)[None].repeat(len(env_ids), 1)
        pos += 0.25 * rand_floats[:, 5:5 + self.num_allegro_hand_dofs]
        pos = tensor_clamp(pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)

        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(hand_indices), len(env_ids)
            )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_reward(self, actions):
        """Check grasp validity based on contacts and object height."""
        def list_intersect(li, hash_num):
            # 17 is the object index (after 17 LEAP hand rigid bodies: 0-16)
            # LEAP fingertip indices: 4 (index), 8 (middle), 12 (ring), 16 (thumb)
            obj_id = 17
            query_list = [obj_id * hash_num + 4, obj_id * hash_num + 8,
                         obj_id * hash_num + 12, obj_id * hash_num + 16]
            return len(np.intersect1d(query_list, li))

        assert self.device == 'cpu', "Grasp generation requires CPU pipeline for contact info"

        # Get contacts for each environment
        contacts = [self.gym.get_env_rigid_contacts(env) for env in self.envs]
        contact_list = [
            list_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000)
            for contact in contacts
        ]
        contact_condition = to_torch(contact_list, device=self.device)

        # Get object and fingertip positions
        obj_pos = self.rigid_body_states[:, [-1], :3]
        finger_pos = self.rigid_body_states[:, [4, 8, 12, 16], :3]

        # Grasp validity conditions (tightened for better quality):
        # 1) At least 3 fingertips within 0.08m of object
        finger_dists = torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1))
        cond1 = (finger_dists < 0.08).sum(-1) >= 3
        # 2) At least 3 fingers in contact with object
        cond2 = contact_condition >= 3
        # 3) Object has not fallen below threshold
        cond3 = torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold)
        # 4) Object is nearly stationary (low velocity)
        obj_vel = torch.norm(self.root_state_tensor[self.object_indices, 7:10], dim=-1)
        cond4 = obj_vel < 0.1

        cond = cond1.float() * cond2.float() * cond3.float() * cond4.float()

        # Debug output every 50 steps
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        if self._debug_counter % 50 == 0:
            print(f'[DEBUG] obj_z={obj_pos[0, -1, -1].item():.3f}, vel={obj_vel[0].item():.3f}')
            print(f'[DEBUG] finger dists: {finger_dists[0].cpu().numpy()}')
            print(f'[DEBUG] contacts={contact_condition[0].item():.0f}, cond1={cond1[0].item()}, cond2={cond2[0].item()}, cond3={cond3[0].item()}, cond4={cond4[0].item()}')

        # Reset environments that don't meet conditions
        self.reset_buf[cond < 1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
