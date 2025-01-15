import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations


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

# Naive Loader (w/o retarget)
class AMPLoader:
    def __init__(
            self,
            device,
            time_between_frames,
            motion_files=glob.glob('datasets/motions_AMASS/*/*/*'),
            preload_transitions=False,
            num_preload_transitions=1000000,
            ):
        """Expert dataset provides AMP observations from AMASS dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.default_fps = 120 # 120  # Hz
        self.frame_delta = 1 # 1
        self.dt = 1 / self.default_fps
        self.amp_dim = 10
        
        # Values to store for each trajectory.
        self.trajs = []
        self.traj_names = []
        self.traj_idxs = []
        # self.traj_weights = []
        self.traj_lens = []
        self.traj_num_frames = []

        for i, motion_file in enumerate(motion_files):
            data = np.load(motion_file)
            all_poses = data['poses']
    
            poses_left_hip = all_poses[:, 1*3:1*3+3]
            poses_right_hip = all_poses[:, 2*3:2*3+3]
            poses_left_knee = all_poses[:, 4*3:4*3+3]
            poses_right_knee = all_poses[:, 5*3:5*3+3]
            poses_left_ankle = all_poses[:, 7*3:7*3+3]
            poses_right_ankle = all_poses[:, 8*3:8*3+3]
            
            concat_poses = np.concatenate(
                [
                    poses_left_hip[:, 1:2],     # left_hip_yaw
                    poses_left_hip[:, 2:3],     # left_hip_roll
                    poses_left_hip[:, 0:1],     # left_hip_pitch
                    poses_left_knee[:, 0:1],    # left_knee
                    poses_left_ankle[:, 0:1],   # left_ankle
                    
                    poses_right_hip[:, 1:2],    # right_hip_yaw
                    poses_right_hip[:, 2:3],    # right_hip_roll
                    poses_right_hip[:, 0:1],    # right_hip_pitch
                    poses_right_knee[:, 0:1],   # right_knee
                    poses_right_ankle[:, 0:1],  # right_ankle
                ],
                axis=1
            )
            concat_poses[:,3] = concat_poses[:,3].clip(min=-0.26)  # left_knee
            concat_poses[:,9] = concat_poses[:,9].clip(min=-0.26)  # right_knee
            concat_poses = concat_poses[::self.frame_delta]

            frame_num, frame_dim = concat_poses.shape
            assert frame_dim == self.amp_dim
            traj_len = frame_num * self.dt

            self.trajs.append(torch.tensor(concat_poses, dtype=torch.float32, device=self.device))
            self.traj_names.append(motion_file.split('/')[-1].split('.')[0])
            self.traj_idxs.append(i)
            # self.traj_weights.append(1.0)
            self.traj_lens.append(traj_len)
            self.traj_num_frames.append(frame_num)

            # print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Use traj weights to sample some trajectories more than others if needed.
        # self.traj_weights = np.array(self.traj_weights) / np.sum(self.traj_weights)
        self.traj_lens = np.array(self.traj_lens)
        self.traj_num_frames = np.array(self.traj_num_frames)

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
        return np.random.choice(self.traj_idxs, size=size, replace=True)
    
    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.dt
        time_samples = self.traj_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.traj_lens[traj_idxs]
        n = self.traj_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.amp_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.amp_dim, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajs[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)
    
    def get_full_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.traj_lens[traj_idx]
        n = self.trajs[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajs[traj_idx][idx_low]
        frame_end = self.trajs[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)
    
    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1
    
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