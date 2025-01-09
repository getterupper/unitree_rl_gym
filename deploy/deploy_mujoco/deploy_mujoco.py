import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def load_target_jt(file, offset):
    one_target_jt = np.load(f"{file}").astype(np.float32)
    target_jt = one_target_jt[np.newaxis, :]
    target_jt += offset

    size = one_target_jt.shape[0]
    return target_jt, size

def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        human_filename = config["human_filename"]
        human_freq = config["human_freq"]

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    # load target joint
    num_envs = 1
    # dt = simulation_dt * control_decimation
    dt = simulation_dt
    target_jt_seq, target_jt_seq_len = load_target_jt(human_filename, default_angles)
    num_target_jt_seq, max_target_jt_seq_len, dim_target_jt = target_jt_seq.shape
    print(f"Loaded target joint trajectories of shape {target_jt_seq.shape}")
    target_jt_i = 0
    target_jt_j = np.zeros(num_envs, dtype=np.int_)
    target_jt_dt = 1 / human_freq
    target_jt_update_steps = target_jt_dt / dt # not necessary integer
    assert(dt <= target_jt_dt)
    target_jt_update_steps_int = sample_int_from_float(target_jt_update_steps)
    target_jt = None
    delayed_obs_target_jt = None
    delayed_obs_target_jt_steps = 0
    delayed_obs_target_jt_steps_int = sample_int_from_float(delayed_obs_target_jt_steps)

    # update target joint
    target_jt = target_jt_seq[target_jt_i, target_jt_j]
    delayed_obs_target_jt = target_jt_seq[target_jt_i, np.maximum(target_jt_j - delayed_obs_target_jt_steps_int, np.array(0))]
    if counter % target_jt_update_steps_int == 0:
        target_jt_j += 1
        jt_eps_end_bool = target_jt_j >= target_jt_seq_len
        target_jt_j = np.where(jt_eps_end_bool, np.zeros_like(target_jt_j), target_jt_j)
        target_jt_update_steps_int = sample_int_from_float(target_jt_update_steps)
        delayed_obs_target_jt_steps_int = sample_int_from_float(delayed_obs_target_jt_steps)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                # period = 0.8
                # count = counter * simulation_dt
                # phase = count % period / period
                # sin_phase = np.sin(2 * np.pi * phase)
                # cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions: 19 + 3 * num_actions] = delayed_obs_target_jt * 1.0  # obs_scales.dof_pos

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
            
            # update target joint
            target_jt = target_jt_seq[target_jt_i, target_jt_j]
            delayed_obs_target_jt = target_jt_seq[target_jt_i, np.maximum(target_jt_j - delayed_obs_target_jt_steps_int, np.array(0))]
            if counter % target_jt_update_steps_int == 0:
                target_jt_j += 1
                jt_eps_end_bool = target_jt_j >= target_jt_seq_len
                target_jt_j = np.where(jt_eps_end_bool, np.zeros_like(target_jt_j), target_jt_j)
                target_jt_update_steps_int = sample_int_from_float(target_jt_update_steps)
                delayed_obs_target_jt_steps_int = sample_int_from_float(delayed_obs_target_jt_steps)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
