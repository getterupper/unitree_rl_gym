import numpy as np

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

def main():
    data_path = "/mnt/c/Users/24050/Desktop/Robotic2025/unitree_rl_gym/legged_gym/data/Walk B15 - Walk turn around_poses.npz"

    data = np.load(data_path)
    print(data.files)
    
    all_poses = data['poses']
    
    poses_left_hip = all_poses[:, 1*3:1*3+3]
    poses_right_hip = all_poses[:, 2*3:2*3+3]
    poses_left_knee = all_poses[:, 4*3:4*3+3]
    poses_right_knee = all_poses[:, 5*3:5*3+3]
    poses_left_ankle = all_poses[:, 7*3:7*3+3]
    poses_right_ankle = all_poses[:, 8*3:8*3+3]
    
    concat_poses = np.concatenate(
        [
            poses_left_hip[:, 1:2],
            poses_left_hip[:, 2:3],
            poses_left_hip[:, 0:1],
            
            poses_left_knee[:, 0:1],
            
            poses_left_ankle[:, 0:1],
            
            
            poses_right_hip[:, 1:2],
            poses_right_hip[:, 2:3],
            poses_right_hip[:, 0:1],
            
            poses_right_knee[:, 0:1],
            
            poses_right_ankle[:, 0:1],
        ],
        axis=1
    )
    frame_num, frame_dim = concat_poses.shape
    print(frame_dim)
    
    concat_poses[:,3] = concat_poses[:,3].clip(min=-0.26) # left knee
    concat_poses[:,9] = concat_poses[:,9].clip(min=-0.26) # right knee
    
    # clip
    delta_t = 1/120
    
    clip_vision = 2
    target_freq = 20
    frame_delta = int((1 / target_freq) / delta_t)
    
    if clip_vision == 1:
        # version 1: only 2 cycle, begin with stand
        concat_poses = concat_poses[48:392:frame_delta]
    elif clip_vision == 2:
        # version 2: repeat 4 cycle, begin with walking moment
        concat_poses = concat_poses[256:402]
        repeat_time = 4
        concat_poses = concat_poses[None,:,:].repeat(repeat_time,axis=0).reshape(-1,frame_dim)
        concat_poses = concat_poses[::frame_delta]
    elif clip_vision == 3:
        # version 3: repeat n cycle, begin with stand
        concat_poses_stand = concat_poses.copy()[:256]
        concat_pose_walk = concat_poses[256:402]
        repeat_time = 30
        concat_pose_walk = concat_pose_walk[None,:,:].repeat(repeat_time,axis=0).reshape(-1,frame_dim)
        concat_poses = np.concatenate((concat_poses_stand, concat_pose_walk), axis=0)
        concat_poses = concat_poses[::frame_delta]

    time_matrix = np.zeros(len(concat_poses))
    delta_time_matrix = np.zeros(len(concat_poses))
    for i in range(len(concat_poses)):
        time_matrix[i] = delta_t * i
        delta_time_matrix[i] = delta_t
    
    np.save(f'h1_test_v{clip_vision}_freq{target_freq}.npy', concat_poses)

if __name__ == "__main__":
    main()