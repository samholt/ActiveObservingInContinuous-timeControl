from __future__ import absolute_import, division, print_function

import copy
from os import path

import numpy as np
import torch

from .base_env import BaseEnv


class CTCancer(BaseEnv):
    """Medically realistic data simulation for small-cell lung cancer based on Geng et al 2017.
    URL: https://www.nature.com/articles/s41598-017-13646-z

    https://arxiv.org/pdf/2206.08311.pdf

    The precise equation for reward:
        -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
        Theta is normalized between -pi and pi. Therefore, the lowest reward is -(pi^2 + 0.1*8^2 + 0.001*2^2) = -16.2736044,
        and the highest reward is 0. In essence, the goal is to remain at zero angle (vertical),
        with the least rotational velocity, and the least effort.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        dt=0.1,
        device="cpu",
        obs_trans=True,
        obs_noise=0.0,
        ts_grid="fixed",
        solver="dopri8",
        friction=False,
        probabilistic=False,
    ):
        name = "cancer"
        state_action_names = ["cancer_volume", "chemo_action", "radio_action"]
        # self.reward_range = [-17.0, 0.0] # for visualization
        super().__init__(
            dt, 2, 2, 2.0, obs_trans, name, state_action_names, device, solver, obs_noise, ts_grid, ac_rew_const=0.001
        )  # 0.001)
        # Params
        self.action_max = 2.0
        # self.nominal_chemo_drug = 5
        self.max_chemo_drug = 5.0

        # self.nominal_radio = 2.0
        self.max_radio = 2.0

        self.chemo_cost_scale = 0.1

        self.use_e_noise = False
        self.time_multiplier = 5.0  # So can use the same time discretization settings across all the other envs

        # Sim params
        if probabilistic:
            self.e_noise = torch.randn(1).to(device)[0] * 0.01
            self.rho = 7e-5 + torch.randn(1).to(device)[0] * 7.23e-3
            self.K = calc_volume(30)
            self.beta_c = 0.028 + torch.randn(1).to(device)[0] * 0.0007
            self.alpha_r = 0.0398 + torch.randn(1).to(device)[0] * 0.168
            self.beta_r = self.alpha_r / 10.0
        else:
            self.e_noise = 0
            # self.rho = 7e-5 + 7.23e-3 * 4 # Good for inducing death
            self.rho = 7e-5 + 7.23e-3 * 2
            self.K = calc_volume(30)
            self.beta_c = 0.028
            self.alpha_r = 0.0398
            self.beta_r = self.alpha_r / 10.0
        self.v_death_thres = calc_volume(13)
        self.reset()

    #################### environment specific ##################

    def torch_transform_states(self, state):
        """Input - [N,n] or [L,N,n]"""
        return state

    def set_state_(self, state):
        self.state = copy.deepcopy(state)
        return self.get_obs()

    def df_du(self, state):
        theta, theta_dot = state[..., 0], state[..., 1]
        m, l = self.mass, self.l
        return torch.stack([theta * 0.0, torch.ones_like(theta_dot) * 3.0 / (m * l**2)], -1)

    #################### override ##################
    def reset(self):
        state = self.np_random.uniform(low=calc_volume(13) * 0.98, high=calc_volume(13) * 0.99, size=(2,)).astype(
            "float32"
        )
        state[1] = 0.0  # Zero drug concentration starting
        self.state = state
        self.time_step = 0
        return self.get_obs()
        # # low, high = np.array([-0.75*np.pi, -1]), np.array([-0.5*np.pi, 1])
        # self.state = self.np_random.uniform(low=low, high=high)
        # self.time_step = 0
        # return self.get_obs()

    def obs2state(self, obs):
        return obs

    def torch_rhs(self, state, action):
        """Input
            state  [N,n]
            action [N,m]
        Main simulation
        state::
        v = cancer_volume
        c = chemo_concentration

        action::
        ca = chemo_action (dosage)
        ra = radio_action (dosage)

        e = noise
        """
        if self.use_e_noise:
            e = torch.rand(1).to(self.d)[0] * 0.1
        else:
            e = self.e_noise
        rho = self.rho
        K = self.K
        beta_c = self.beta_c
        alpha_r = self.alpha_r
        beta_r = self.beta_r

        v, c = state[..., 0], state[..., 1]
        v[v <= 0] = 0
        action = torch.clamp(action, min=-self.action_max, max=self.action_max)
        ca_unshifted, ra_unshifted = action[..., 0], action[..., 1]
        ca = (ca_unshifted / 2.0) * self.max_chemo_drug
        ra = (ra_unshifted / 2.0) * self.max_radio
        ca[ca <= 0] = 0
        ra[ra <= 0] = 0

        # ca = torch.tensor(self.max_chemo_drug, device=self.d)
        # ra = torch.tensor(self.max_radio, device=self.d)
        dc_dt = -c / 2 + ca
        dv_dt = (rho * torch.log(K / v) - beta_c * c - (alpha_r * ra + beta_r * torch.square(ra)) + e) * v
        dv_dt[v == 0] = 0
        dv_dt = dv_dt.nan_to_num(posinf=0, neginf=0)
        return torch.stack([dv_dt * self.time_multiplier, dc_dt * self.time_multiplier], -1)

    def diff_obs_reward_(self, state, exp_reward=False):
        v, c = state[..., 0], state[..., 1]
        v[v <= 0] = 0
        state_reward = -torch.square(calc_diameter(v))
        # state_reward = -torch.abs(v/self.v_death_thres)
        # state_reward = -torch.exp(v)
        # state_reward = -torch.square(v)
        state_reward -= self.chemo_cost_scale * torch.square(c)
        # state_reward -= (v>=self.v_death_thres).float() * 10000
        return state_reward

    def diff_ac_reward_(self, action):
        return -self.ac_rew_const * torch.sum(torch.square((action / self.action_max)), -1)

    def render(self, mode="human", **kwargs):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        try:
            last_act = kwargs["last_act"]
            self.imgtrans.scale = (-last_act / 2, np.abs(last_act) / 2)
        except:
            pass
        return self.viewer.render(return_rgb_array=mode == "rgb_array")


def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0


def calc_diameter(volume):
    return ((volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)) * 2.0
