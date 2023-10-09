import numpy as np
import os
import glob
import json
import yaml

import torch

from isaacgymenvs.utils.torch_jit_utils import to_torch, slerp, quat_to_exp_map, quat_to_angle_axis, normalize_angle


class MotionLoader():

    def __init__(self, motion_file, device):
        self._device = device
        self._load_motions(motion_file)

        self.motion_ids = torch.arange(len(self._motions))

        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        m = self.num_motions()
        motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)

        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len

        return motion_time

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)

        root_pos0 = torch.zeros(n, MotionLoader.POS_SIZE, device=self._device)
        root_pos1 = torch.zeros(n, MotionLoader.POS_SIZE, device=self._device)
        root_rot0 = torch.zeros(n, MotionLoader.ROT_SIZE, device=self._device)
        root_rot1 = torch.zeros(n, MotionLoader.ROT_SIZE, device=self._device)
        joint_pos0 = torch.zeros(n, MotionLoader.JOINT_POS_SIZE, device=self._device)
        joint_pos1 = torch.zeros(n, MotionLoader.JOINT_POS_SIZE, device=self._device)
        local_ee_pos0 = torch.zeros(n, MotionLoader.TAR_TOE_POS_LOCAL_SIZE, device=self._device)
        local_ee_pos1 = torch.zeros(n, MotionLoader.TAR_TOE_POS_LOCAL_SIZE, device=self._device)
        lin_vel0 = torch.zeros(n, MotionLoader.LINEAR_VEL_SIZE, device=self._device)
        lin_vel1 = torch.zeros(n, MotionLoader.LINEAR_VEL_SIZE, device=self._device)
        ang_vel0 = torch.zeros(n, MotionLoader.ANGULAR_VEL_SIZE, device=self._device)
        ang_vel1 = torch.zeros(n, MotionLoader.ANGULAR_VEL_SIZE, device=self._device)
        joint_vel0 = torch.zeros(n, MotionLoader.JOINT_VEL_SIZE, device=self._device)
        joint_vel1 = torch.zeros(n, MotionLoader.JOINT_VEL_SIZE, device=self._device)

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            curr_motion = self._motions[uid]

            root_pos0[ids, :] = MotionLoader.get_root_pos_batch(curr_motion[frame_idx0[ids]])
            root_pos1[ids, :] = MotionLoader.get_root_pos_batch(curr_motion[frame_idx1[ids]])
            root_rot0[ids, :] = MotionLoader.get_root_rot_batch(curr_motion[frame_idx0[ids]])
            root_rot1[ids, :] = MotionLoader.get_root_rot_batch(curr_motion[frame_idx1[ids]])
            joint_pos0[ids, :] = MotionLoader.get_joint_pose_batch(curr_motion[frame_idx0[ids]])
            joint_pos1[ids, :] = MotionLoader.get_joint_pose_batch(curr_motion[frame_idx1[ids]])
            local_ee_pos0[ids, :] = MotionLoader.get_tar_toe_pos_local_batch(curr_motion[frame_idx0[ids]])
            local_ee_pos1[ids, :] = MotionLoader.get_tar_toe_pos_local_batch(curr_motion[frame_idx1[ids]])
            lin_vel0[ids, :] = MotionLoader.get_linear_vel_batch(curr_motion[frame_idx0[ids]])
            lin_vel1[ids, :] = MotionLoader.get_linear_vel_batch(curr_motion[frame_idx1[ids]])
            ang_vel0[ids, :] = MotionLoader.get_angular_vel_batch(curr_motion[frame_idx0[ids]])
            ang_vel1[ids, :] = MotionLoader.get_angular_vel_batch(curr_motion[frame_idx1[ids]])
            joint_vel0[ids, :] = MotionLoader.get_joint_vel_batch(curr_motion[frame_idx0[ids]])
            joint_vel1[ids, :] = MotionLoader.get_joint_vel_batch(curr_motion[frame_idx1[ids]])

        blend = to_torch(np.expand_dims(blend, axis=-1), device=self._device)

        root_pos = self.linear_slerp(root_pos0, root_pos1, blend)
        root_rot = self.quaternion_slerp(root_rot0, root_rot1, blend)
        joint_pos = self.linear_slerp(joint_pos0, joint_pos1, blend)
        ee_pos = self.linear_slerp(local_ee_pos0, local_ee_pos1, blend)
        lin_vel = self.linear_slerp(lin_vel0, lin_vel1, blend)
        ang_vel = self.linear_slerp(ang_vel0, ang_vel1, blend)
        joint_vel = self.linear_slerp(joint_vel0, joint_vel1, blend)

        return root_pos, root_rot, joint_pos, ee_pos, lin_vel, ang_vel, joint_vel

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            with open(curr_file, "r") as file:
                motion_json = json.load(file)
                curr_full_motions = np.array(motion_json["Frames"])
                curr_dt = float(motion_json["FrameDuration"])
                motion_fps = 1.0 / curr_dt

                num_frames = curr_full_motions.shape[0]
                curr_len = curr_dt * (num_frames - 1)  # time length

                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)

                for f_i in range(num_frames):
                    curr_full_motions[f_i] = self.normalize_and_standardize_quat(curr_full_motions[f_i])

                self._motions.append(torch.tensor(curr_full_motions, dtype=torch.float32, device=self._device))
                self._motion_lengths.append(curr_len)

                curr_weight = motion_weights[f]
                self._motion_weights.append(curr_weight)
                self._motion_files.append(curr_file)

        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert (curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = np.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).astype(int)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def linear_slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def quaternion_slerp(self, q0, q1, fraction, spin=0, shortestpath=True):
        """Batch quaternion spherical linear interpolation."""

        _EPS = np.finfo(float).eps * 4.0
        out = torch.zeros_like(q0)

        zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
        ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
        out[zero_mask] = q0[zero_mask]
        out[ones_mask] = q1[ones_mask]

        d = torch.sum(q0 * q1, dim=-1, keepdim=True)
        dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
        out[dist_mask] = q0[dist_mask]

        if shortestpath:
            d_old = torch.clone(d)
            d = torch.where(d_old < 0, -d, d)
            q1 = torch.where(d_old < 0, -q1, q1)

        angle = torch.acos(d) + spin * torch.pi
        angle_mask = (torch.abs(angle) < _EPS).squeeze()
        out[angle_mask] = q0[angle_mask]

        final_mask = torch.logical_or(zero_mask, ones_mask)
        final_mask = torch.logical_or(final_mask, dist_mask)
        final_mask = torch.logical_or(final_mask, angle_mask)
        final_mask = torch.logical_not(final_mask)

        isin = 1.0 / angle
        q0 *= torch.sin((1.0 - fraction) * angle) * isin
        q1 *= torch.sin(fraction * angle) * isin
        q0 += q1
        out[final_mask] = q0[final_mask]
        return out

    def normalize_and_standardize_quat(self, motion_data):
        root_rot = MotionLoader.get_root_rot(motion_data)
        root_rot = root_rot / np.linalg.norm(root_rot)  # Normalizes the quaternion to length 1
        if root_rot[-1] < 0:
            root_rot = -root_rot
        motion_data[MotionLoader.POS_SIZE:(MotionLoader.POS_SIZE + MotionLoader.ROT_SIZE)] = root_rot

        return motion_data

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 6
    TAR_TOE_POS_LOCAL_SIZE = 6
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 6
    TAR_TOE_VEL_LOCAL_SIZE = 6

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    def get_root_pos(pose):
        return pose[MotionLoader.ROOT_POS_START_IDX:MotionLoader.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoader.ROOT_POS_START_IDX:MotionLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoader.ROOT_ROT_START_IDX:MotionLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoader.ROOT_ROT_START_IDX:MotionLoader.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoader.JOINT_POSE_START_IDX:MotionLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoader.JOINT_POSE_START_IDX:MotionLoader.JOINT_POSE_END_IDX]

    def get_tar_toe_pos_local(pose):
        return pose[MotionLoader.TAR_TOE_POS_LOCAL_START_IDX:MotionLoader.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, MotionLoader.TAR_TOE_POS_LOCAL_START_IDX:MotionLoader.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoader.LINEAR_VEL_START_IDX:MotionLoader.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoader.LINEAR_VEL_START_IDX:MotionLoader.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoader.ANGULAR_VEL_START_IDX:MotionLoader.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoader.ANGULAR_VEL_START_IDX:MotionLoader.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[MotionLoader.JOINT_VEL_START_IDX:MotionLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoader.JOINT_VEL_START_IDX:MotionLoader.JOINT_VEL_END_IDX]

    def get_tar_toe_vel_local(pose):
        return pose[MotionLoader.TAR_TOE_VEL_LOCAL_START_IDX:MotionLoader.TAR_TOE_VEL_LOCAL_END_IDX]

    def get_tar_toe_vel_local_batch(poses):
        return poses[:, MotionLoader.TAR_TOE_VEL_LOCAL_START_IDX:MotionLoader.TAR_TOE_VEL_LOCAL_END_IDX]
