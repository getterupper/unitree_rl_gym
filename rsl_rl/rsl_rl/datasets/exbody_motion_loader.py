import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from .poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from .poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
from .utils import torch_utils

import pdb
import yaml

'''
     0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_index1',
    23: 'left_index2',
    24: 'left_index3',
    25: 'left_middle1',
    26: 'left_middle2',
    27: 'left_middle3',
    28: 'left_pinky1',
    29: 'left_pinky2',
    30: 'left_pinky3',
    31: 'left_ring1',
    32: 'left_ring2',
    33: 'left_ring3',
    34: 'left_thumb1',
    35: 'left_thumb2',
    36: 'left_thumb3',
    37: 'right_index1',
    38: 'right_index2',
    39: 'right_index3',
    40: 'right_middle1',
    41: 'right_middle2',
    42: 'right_middle3',
    43: 'right_pinky1',
    44: 'right_pinky2',
    45: 'right_pinky3',
    46: 'right_ring1',
    47: 'right_ring2',
    48: 'right_ring3',
    49: 'right_thumb1',
    50: 'right_thumb2',
    51: 'right_thumb3'
'''

class ExBodyAMPLoader:
    def __init__(
            self,
            device,
            time_between_frames,
            motion_files=glob.glob('datasets/motions_Mocap/*/*'),
            preload_transitions=False,
            num_preload_transitions=1000000,
            save_poses=False,
            ):
        """Expert dataset provides AMP observations from CMU Mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.default_fps = 60  # Hz
        self.dt = 1 / self.default_fps

        self.dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                             4, 5, 6,
                             7,       # Torso
                             8, 9, 10, # Shoulder, Elbow, Hand
                             11, 12, 13]  # 13
        self.dof_offsets = [0, 3, 4, 5, 8, 9, 10, 11, 
                            14, 15, 16, 19, 20, 21]  # 14
        self.num_dof = self.dof_offsets[-1]
        
        self.file_infos_path = 'rsl_rl/rsl_rl/datasets/poselib/data/configs/motions_autogen_all_no_run_jump.yaml'
        with open(self.file_infos_path, 'r', encoding='utf-8') as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.file_infos = result['motions']

        self.dof_indices_sim = torch.tensor([0, 1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 18], device=self.device, dtype=torch.long)
        self.dof_indices_motion = torch.tensor([2, 0, 1, 7, 5, 6, 12, 11, 13, 17, 16, 18], device=self.device, dtype=torch.long)

        self.amp_dim = 33  # 3 + 4 + 3 + 3 + 10 + 10
        self.part_amp_dim = 20

        self.motion_infos = []
        self.motion_idxs = []
        self.motion_lens = []
        self.motion_num_frames = []

        for i, motion_file in enumerate(motion_files):
            name = motion_file.split('/')[-1].split('.')[0]
            curr_info = self.file_infos[name]
            if 'walk' not in curr_info['description']:
                continue
            
            curr_motion = SkeletonMotion.from_file(motion_file)
            
            motion_fps = curr_motion.fps
            assert motion_fps == self.default_fps

            frame_num = curr_motion.tensor.shape[0]
            motion_len = frame_num * self.dt
            
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            curr_dof_pos = self._compute_motion_dof_pos(curr_motion)
            curr_motion.dof_pos = curr_dof_pos

            curr_motion.tensor = curr_motion.tensor.to(self.device)
            curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self.device)
            curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self.device)
            curr_motion._rotation = curr_motion._rotation.to(self.device)
            
            self.motion_infos.append(curr_motion)
            self.motion_idxs.append(i)
            self.motion_lens.append(motion_len)
            self.motion_num_frames.append(frame_num)

            # save_poses = True
            if save_poses:
                name = motion_file.split('/')[-1].split('.')[0]
                time_matrix = np.zeros(frame_num)
                delta_time_matrix = np.zeros(frame_num)
                stance_mask_left = np.zeros(frame_num)
                stance_mask_right = np.zeros(frame_num)
                for j in range(frame_num):
                    time_matrix[j] = self.dt * j
                    delta_time_matrix[j] = self.dt

                curr_dof_pos = reindex_motion_dof(curr_dof_pos, self.dof_indices_sim, self.dof_indices_motion)

                joints = np.concatenate((time_matrix.reshape(-1, 1), delta_time_matrix.reshape(-1, 1), curr_dof_pos.cpu().numpy()[..., :10], stance_mask_left.reshape(-1, 1), stance_mask_right.reshape(-1, 1)), axis=1)
                J_head = np.array(['time', 'delta_t',
                                'left_leg_pitch_joint', 'left_leg_roll_joint', 'left_leg_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                                'right_leg_pitch_joint', 'right_leg_roll_joint', 'right_leg_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
                                'stance_mask_left', 'stance_mask_right'])
                J_header_string = ','.join(J_head)
                np.savetxt(f'./tmp/{name}_60fps.csv', joints, delimiter=',', fmt='%.8e', header=J_header_string, comments='')

        self.motion_lens = np.array(self.motion_lens)
        self.motion_num_frames = np.array(self.motion_num_frames)

        self.gts = [m.global_translation for m in self.motion_infos]
        self.grs = [m.global_rotation for m in self.motion_infos]
        self.lrs = [m.local_rotation for m in self.motion_infos]
        self.grvs = [m.global_root_velocity for m in self.motion_infos]
        self.gravs = [m.global_root_angular_velocity for m in self.motion_infos]
        self.dvs = [m.dof_vels for m in self.motion_infos]
        self.dps = [m.dof_pos for m in self.motion_infos]

        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

    def get_full_frame_batch(self, num_frames):
        traj_idxs = self.traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.motion_idxs, size=size, replace=True)
    
    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.dt
        time_samples = self.motion_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)
    
    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.motion_lens[traj_idxs]
        n = self.motion_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        all_frame_info = torch.zeros(len(traj_idxs), self.part_amp_dim, device=self.device)
        
        for traj_idx in set(traj_idxs):
            traj_mask = traj_idxs == traj_idx

            # root_pos_low = self.gts[traj_idx][idx_low, 0]
            # root_pos_high = self.gts[traj_idx][idx_high, 0]
            # root_pos = (1.0 - blend) * root_pos_low + blend * root_pos_high

            # root_rot_low = self.grs[traj_idx][idx_low, 0]
            # root_rot_high = self.grs[traj_idx][idx_high, 0]
            # root_rot = torch_utils.slerp(root_rot_low, root_rot_high, blend)

            # root_vel = self.grvs[traj_idx][idx_low]
            # root_ang_vel = self.gravs[traj_idx][idx_low]
            
            # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
            # local_root_vel = quat_rotate(heading_rot, root_vel)
            # local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

            # dof_pos_low = self.dps[traj_idx][idx_low, :10]
            # dof_pos_high = self.dps[traj_idx][idx_high, :10]
            # dof_pos = (1 - blend) * dof_pos_low + blend * dof_pos_high
            local_rot_low = self.lrs[traj_idx][idx_low[traj_mask]]
            local_rot_high = self.lrs[traj_idx][idx_high[traj_mask]]
            local_rot = torch_utils.slerp(local_rot_low, local_rot_high, torch.unsqueeze(blend[traj_mask], axis=-1))
            dof_pos = self._local_rotation_to_dof(local_rot)
            dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion)
            dof_pos = dof_pos[..., :10]

            dof_vel = self.dvs[traj_idx][idx_low[traj_mask]]
            dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion)
            dof_vel = dof_vel[..., :10]

            curr_traj_info = torch.cat([dof_pos, dof_vel], dim=-1).float().to(self.device)

            all_frame_info[traj_mask] = curr_traj_info
        
        return all_frame_info
    
    def get_full_frame_batch_rsi(self, num_frames):
        traj_idxs = self.traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch_rsi(traj_idxs, times)
    
    def get_full_frame_at_time_batch_rsi(self, traj_idxs, times):
        p = times / self.motion_lens[traj_idxs]
        n = self.motion_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        all_frame_info = torch.zeros(len(traj_idxs), self.amp_dim, device=self.device)
        
        for traj_idx in set(traj_idxs):
            traj_mask = traj_idxs == traj_idx

            root_pos_low = self.gts[traj_idx][idx_low[traj_mask], 0]
            root_pos_high = self.gts[traj_idx][idx_high[traj_mask], 0]
            root_pos = (1.0 - blend[traj_mask]) * root_pos_low + blend[traj_mask] * root_pos_high

            root_rot_low = self.grs[traj_idx][idx_low[traj_mask], 0]
            root_rot_high = self.grs[traj_idx][idx_high[traj_mask], 0]
            root_rot = torch_utils.slerp(root_rot_low, root_rot_high, blend[traj_mask])

            root_vel = self.grvs[traj_idx][idx_low[traj_mask]]
            root_ang_vel = self.gravs[traj_idx][idx_low[traj_mask]]
            
            heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
            local_root_vel = quat_rotate(heading_rot, root_vel)
            local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

            # dof_pos_low = self.dps[traj_idx][idx_low, :10]
            # dof_pos_high = self.dps[traj_idx][idx_high, :10]
            # dof_pos = (1 - blend) * dof_pos_low + blend * dof_pos_high
            local_rot_low = self.lrs[traj_idx][idx_low[traj_mask]]
            local_rot_high = self.lrs[traj_idx][idx_high[traj_mask]]
            local_rot = torch_utils.slerp(local_rot_low, local_rot_high, torch.unsqueeze(blend[traj_mask], axis=-1))
            dof_pos = self._local_rotation_to_dof(local_rot)
            dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion)
            dof_pos = dof_pos[..., :10]

            dof_vel = self.dvs[traj_idx][idx_low[traj_mask]]
            dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion)
            dof_vel = dof_vel[..., :10]

            curr_traj_info = torch.cat([root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel], dim=-1).float().to(self.device)

            all_frame_info[traj_mask] = curr_traj_info
        
        return all_frame_info
    
    def get_full_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.traj_lens[traj_idx]
        n = self.trajs[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        blend = p * n - idx_low

        # root_pos_low = self.gts[traj_idx][idx_low, 0]
        # root_pos_high = self.gts[traj_idx][idx_high, 0]
        # root_pos = (1.0 - blend) * root_pos_low + blend * root_pos_high

        # root_rot_low = self.grs[traj_idx][idx_low, 0]
        # root_rot_high = self.grs[traj_idx][idx_high, 0]
        # root_rot = torch_utils.slerp(root_rot_low, root_rot_high, blend)

        # root_vel = self.grvs[traj_idx][idx_low]
        # root_ang_vel = self.gravs[traj_idx][idx_low]

        # dof_pos_low = self.dps[traj_idx][idx_low, :10]
        # dof_pos_high = self.dps[traj_idx][idx_high, :10]
        # dof_pos = (1 - blend) * dof_pos_low + blend * dof_pos_high
        local_rot_low = self.lrs[traj_idx][idx_low]
        local_rot_high = self.lrs[traj_idx][idx_high]
        local_rot = torch_utils.slerp(local_rot_low, local_rot_high, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)
        dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion)
        dof_pos = dof_pos[..., :10]

        dof_vel = self.dvs[traj_idx][idx_low]
        dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion)
        dof_vel = dof_vel[..., :10]

        curr_traj_info = torch.cat([dof_pos, dof_vel], dim=-1).float().to(self.device)
        
        return curr_traj_info

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = self.dt
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)
        return dof_vels

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self.dof_body_ids
        dof_offsets = self.dof_offsets

        dof_vel = torch.zeros([self.num_dof], device=self.device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            # print(joint_offset, joint_offset+joint_size)

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel
    
    def _compute_motion_dof_pos(self, motion):
        num_frames = motion.tensor.shape[0]
        dof_pos = []

        for f in range(num_frames):
            local_rot = motion.local_rotation[f].unsqueeze(0)
            frame_dof_pos = self._local_rotation_to_dof(local_rot)
            frame_dof_pos = frame_dof_pos
            dof_pos.append(frame_dof_pos)

        dof_pos = torch.cat(dof_pos, dim=0)
        return dof_pos
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = self.dof_body_ids
        dof_offsets = self.dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self.num_dof), dtype=torch.float, device=self.device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos
    
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, ...]
                s_next = self.preloaded_s_next[idxs, ...]
            else:
                s, s_next = [], []
                traj_idxs = self.traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_full_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_full_frame_at_time(traj_idx, frame_time + self.time_between_frames))
                
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next
    
@torch.jit.script
def reindex_motion_dof(dof, indices_sim, indices_motion):
    dof = dof.clone()
    dof[:, indices_sim] = dof[:, indices_motion]
    return dof