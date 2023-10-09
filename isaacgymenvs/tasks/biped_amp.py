from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymtorch

from isaacgymenvs.tasks.amp.biped_amp_base import BipedAMPBase
from isaacgymenvs.tasks.amp.utils_amp.motion_loader import MotionLoader

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, calc_heading_quat_inv, quat_to_tan_norm, \
    my_quat_rotate

NUM_AMP_OBS_PER_STEP = 6 + 3 + 3 + 6 + 1  # [dof_pos, root_vel, root_ang_vel, dof_vel, root_h]


class BipedAMP(BipedAMPBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = BipedAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert (self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(cfg=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # load motion data
        motion_file = cfg['env'].get('motion_file', "biped_locomotion.yaml")
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "../../assets/amp/biped_motions/" + motion_file)
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP

        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf,
                                         np.ones(self.num_amp_obs) * np.Inf)  # infnite value space

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP),
                                        device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        return

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self.num_amp_obs

    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_loader.sample_motions(num_samples)

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert (self._amp_obs_demo_buf.shape[0] == num_samples)

        motion_times0 = self._motion_loader.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()

        root_pos, root_rot, joint_pos, ee_pos, lin_vel, ang_vel, joint_vel \
            = self._motion_loader.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, lin_vel, ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, lin_vel, ang_vel, joint_pos, joint_vel)

        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP),
                                             device=self.device, dtype=torch.float)
        return

    def _load_motion(self, motion_file):
        self._motion_loader = MotionLoader(motion_file=motion_file, device=self.device)
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == BipedAMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == BipedAMP.StateInit.Start
              or self._state_init == BipedAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == BipedAMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        return

    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_loader.sample_motions(num_envs)

        if (self._state_init == BipedAMP.StateInit.Random
                or self._state_init == BipedAMP.StateInit.Hybrid):
            motion_times = self._motion_loader.sample_time(motion_ids)
        elif (self._state_init == BipedAMP.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, joint_pos, ee_pos, lin_vel, ang_vel, joint_vel \
            = self._motion_loader.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=joint_pos,
                            root_vel=lin_vel,
                            root_ang_vel=ang_vel,
                            dof_vel=joint_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, joint_pos, ee_pos, lin_vel, ang_vel, joint_vel \
            = self._motion_loader.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, lin_vel, ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, lin_vel, ang_vel, joint_pos, joint_vel)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self.root_states, self.base_lin_vel,
                                                               self.base_ang_vel, self.dof_pos, self.dof_vel)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self.root_states[env_ids],
                                                                     self.base_lin_vel[env_ids],
                                                                     self.base_ang_vel[env_ids],
                                                                     self.dof_pos[env_ids], self.dof_vel[env_ids])

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def build_amp_observations(root_states, base_lin_vel, base_ang_vel, dof_pos, dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]

    obs = torch.cat((dof_pos, dof_vel, base_lin_vel, base_ang_vel, root_h), dim=-1)
    return obs
