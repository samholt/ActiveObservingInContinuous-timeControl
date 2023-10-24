import torch
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal
import functools
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray

class MPPIActiveObserving():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, running_cost, nx, noise_sigma, 
                 cost_var_from_state_var=None,
                 num_samples=100, horizon=15, device="cuda:0",
                 terminal_state_cost=None,
                 observing_var_threshold=1.0,
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 rollout_samples=1, # Ensemble size
                 rollout_var_cost=0,
                 rollout_var_discount=0.95,
                 dt_simulation=0.01,
                 dt=0.05,
                 sampling_policy='discrete_planning',
                 continuous_time_threshold=0.5,
                 observing_cost=1.0,
                 sample_null_action=False,
                 observing_fixed_frequency=1,
                discrete_planning=False,
                discrete_interval=1,
                limit_actions_to_only_positive=False,
                fixed_continuous_planning_steps=None,
                debug_mode_return_full_cost_std=False,
                debug_mode_cp_return_continuous_reward_unc=False,
                 noise_abs_cost=False):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.d = device
        self.dt_simulation = dt_simulation
        if discrete_planning:
            dt_plan = dt_simulation * discrete_interval
        else:
            dt_plan = dt
        self.discrete_planning = discrete_planning
        self.discrete_interval = discrete_interval
        self.limit_actions_to_only_positive = limit_actions_to_only_positive
        self.continuous_time_interval = max(int(continuous_time_threshold * discrete_interval),1)
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        self.dt = dt_plan
        self.observing_cost = observing_cost # Hyperparameter to be tuned
        self.observing_var_threshold = observing_var_threshold # Hyperparameter to be tuned
        self.observing_fixed_frequency = observing_fixed_frequency

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states_mu = None
        self.states_var = None
        self.actions = None

        self.sampling_policy = sampling_policy
        self.cost_var_from_state_var = cost_var_from_state_var

        self.previous_step = 0
        self.fixed_continuous_planning_steps = fixed_continuous_planning_steps
        self.debug_mode_return_full_cost_std = debug_mode_return_full_cost_std
        self.debug_mode_cp_return_continuous_reward_unc = debug_mode_cp_return_continuous_reward_unc

    def _dynamics(self, state, u, ts_pred, return_var=True):
        if self.limit_actions_to_only_positive:
            u[u<=0] = 0
        return self.F(state, u, ts_pred, return_var=return_var)

    def _cost_var_from_state_var(self, state_var):
        if not self.cost_var_from_state_var is None:
            return self.cost_var_from_state_var(state_var)
        else:
            return state_var.sum()

        

    # @handle_batch_input
    def _running_cost(self, state, u):
        return self.running_cost(state, u)


    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state_mu = self.state
        else:
            state_mu = self.state.view(1, -1).repeat(K, 1)

        logger.debug(f'state: {state_mu.shape}')

        states_mu = []
        # states_var = []
        actions = []
        perturbed_actions = self.u_scale * perturbed_actions
        ts_pred = torch.tensor(self.dt, device=self.d, dtype=self.dtype).view(1, 1).repeat(K, 1)

        for t in range(T):
            u = perturbed_actions[:,t,:]
            state_mu, _ = self._dynamics(state_mu, u, ts_pred, return_var=False)
            c = self._running_cost(state_mu, u)
            cost_samples += c
            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

            # Save total states/actions
            states_mu.append(state_mu)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states_mu = torch.stack(states_mu, dim=-2)
        logger.debug(f'states: {states_mu.shape}')

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states_mu, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)
        cost_total += cost_var * self.rollout_var_cost
        logger.debug(f'{cost_total.shape} | {states_mu.shape} | {actions.shape}')
        return cost_total, states_mu, actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T)) # K x T x nu
        self.perturbed_action = self.U + self.noise
        if self.sample_null_action:
            self.perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action * self.u_scale)
        self.perturbed_action /= self.u_scale
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * self.noise @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nominal trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv # Like original paper
        logger.debug(f'action_cost: {action_cost.shape}')

        self.cost_total, self.states_mu, self.actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions /= self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2)) # wonder if can remove?
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            action = torch.clamp(action, min=self.u_min, max=self.u_max)
        return action

    def get_rollouts(self, state, num_rollouts=1):
        """
            :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
            :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                                 dynamics
            :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device)
        states[:, 0] = state
        ts_pred = torch.tensor(self.dt, device=self.d, dtype=self.dtype).view(1, 1).repeat(num_rollouts, 1)
        for t in range(T):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
                                              self.u_scale * self.U[t].view(num_rollouts, -1), ts_pred)
        return states[:, 1:]


    def command(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        self.U = torch.zeros_like(self.U)

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        assert not torch.isnan(state).any(), "Nan detected in state"

        cost_total = self._compute_total_cost_batch()
        logger.debug(f'cost_total: {cost_total.shape}')

        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        for t in range(self.T):
            self.U[t] += torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)


        # Calculate the state estimate of the reward here, then use that for planning etc.
        if self.debug_mode_cp_return_continuous_reward_unc and self.sampling_policy == 'continuous_planning':
            # Monte Carlo Simulation of latest reward variance
            L = self.K * 10
            ts_pred = torch.tensor(self.dt, device=self.d, dtype=self.dtype).view(1, 1).repeat(L, 1)
            ts_pred_increment = torch.arange(self.dt_simulation, self.dt, self.dt_simulation, device=self.d, dtype=self.dtype).repeat_interleave(L).view(-1,1)
            cost_var = torch.zeros_like(cost_total)
            if self.state.shape == (L, self.nx):
                state_mu = self.state
            else:
                state_mu = self.state.view(1, -1).repeat(L, 1)
            state_mu_in = state_mu
            costs_std = []
            costs_std.append(torch.tensor(0, device=self.d, dtype=self.dtype).view(1))
            same_actions = self.U.unsqueeze(0).repeat(L, 1, 1)
            for t in range(self.T):
                u = same_actions[:,t,:]
                # Core parts
                state_mu_pred, state_var_pred = self._dynamics(state_mu_in, u, ts_pred, return_var=True)
                state_mu_final = state_mu_pred + torch.normal(0, 1, size=state_mu_pred.shape).to(self.d) * torch.sqrt(state_var_pred)
                c = self._running_cost(state_mu_final, u)
                # Intermediate states
                intermediate_state_count = self.discrete_interval - 1
                state_mu_pred_increment, state_var_pred_increment = self._dynamics(state_mu_in.repeat(intermediate_state_count, 1), u.repeat(intermediate_state_count, 1), ts_pred_increment, return_var=True)
                state_mu_increment = state_mu_pred_increment + torch.normal(0, 1, size=state_mu_pred_increment.shape).to(self.d) * torch.sqrt(state_var_pred_increment)
                c_increment = self._running_cost(state_mu_increment, u.repeat(intermediate_state_count, 1))
                inter_c_stds = c_increment.view(intermediate_state_count, -1).std(dim=1)
                costs_std.append(torch.cat((inter_c_stds, c.std().view(1))))
                state_mu_in = state_mu_final
            # States is K x T x nx
            costs_std_continuous = torch.cat(costs_std)[1:]
            stats = {'costs_std_median': costs_std_continuous.median().item(), 'costs_std_mean': costs_std_continuous.mean().item(), 'costs_std_max': costs_std_continuous.max().item()}
            if self.debug_mode_return_full_cost_std:
                return torch.cat(costs_std).cpu()
        elif self.sampling_policy == 'active_observing_control':
            # Monte Carlo Simulation of latest reward variance
            L = self.K * 10
            ts_pred = torch.tensor(self.dt, device=self.d, dtype=self.dtype).view(1, 1).repeat(L, 1)
            ts_pred_increment = torch.arange(self.dt_simulation, self.dt, self.dt_simulation, device=self.d, dtype=self.dtype).repeat_interleave(L).view(-1,1)
            cost_var = torch.zeros_like(cost_total)
            if self.state.shape == (L, self.nx):
                state_mu = self.state
            else:
                state_mu = self.state.view(1, -1).repeat(L, 1)
            state_mu_in = state_mu
            costs_std = []
            costs_std.append(torch.tensor(0, device=self.d, dtype=self.dtype).view(1))
            same_actions = self.U.unsqueeze(0).repeat(L, 1, 1)
            select_actions_up_to = self.T * self.discrete_interval # Initial default value
            for t in range(self.T):
                u = same_actions[:,t,:]
                # Core parts
                state_mu_pred, state_var_pred = self._dynamics(state_mu_in, u, ts_pred, return_var=True)
                state_mu_final = state_mu_pred + torch.normal(0, 1, size=state_mu_pred.shape).to(self.d) * torch.sqrt(state_var_pred)
                c = self._running_cost(state_mu_final, u)
                if c.std() >= self.observing_var_threshold:
                    t_upper = ts_pred.view(-1)[0]
                    t_lower = torch.tensor(0.0).to(self.d)
                    while (t_upper - t_lower) > self.dt_simulation:
                        t_mid = (t_upper + t_lower) / 2.0
                        state_mu_pred_increment, state_var_pred_increment = self._dynamics(state_mu_in, u, torch.ones_like(ts_pred) * t_mid, return_var=True)
                        state_mu_increment = state_mu_pred_increment + torch.normal(0, 1, size=state_mu_pred_increment.shape).to(self.d) * torch.sqrt(state_var_pred_increment)
                        c_increment = self._running_cost(state_mu_increment, u)
                        if c_increment.std() >= self.observing_var_threshold:
                            t_upper = t_mid
                        else:
                            t_lower = t_mid
                    select_actions_up_to = t * self.discrete_interval + torch.floor((t_mid / ts_pred.view(-1)[0]) * self.discrete_interval).int().item()
                    break
                state_mu_in = state_mu_final
            stats = {}
        else:
            # Monte Carlo Simulation of latest reward variance
            L = self.K * 10
            ts_pred = torch.tensor(self.dt, device=self.d, dtype=self.dtype).view(1, 1).repeat(L, 1)
            cost_var = torch.zeros_like(cost_total)
            if self.state.shape == (L, self.nx):
                state_mu = self.state
            else:
                state_mu = self.state.view(1, -1).repeat(L, 1)
            states_mu = []
            states_var = []
            costs = []
            same_actions = self.U.unsqueeze(0).repeat(L, 1, 1)
            for t in range(self.T):
                u = same_actions[:,t,:]
                state_mu, state_var = self._dynamics(state_mu, u, ts_pred, return_var=True)
                state_mu = state_mu + torch.normal(0, 1, size=state_mu.shape).to(self.d) * torch.sqrt(state_var)
                c = self._running_cost(state_mu, u)
                if self.M > 1: # Untested, however should underperform - MPPI with uncertaintity paper
                    cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

                # Save total states/actions
                costs.append(c)
                states_mu.append(state_mu)
                states_var.append(state_var)

            # States is K x T x nx
            states_mu = torch.stack(states_mu, dim=-2)
            states_var = torch.stack(states_var, dim=-2)
            costs = torch.stack(costs, dim=-2)
            costs_std_discrete = torch.cat((torch.tensor(0, device=self.d, dtype=self.dtype).view(1), costs.std(dim=1)))[1:]
            stats = {'costs_std_median': costs_std_discrete.median().item(), 'costs_std_mean': costs_std_discrete.mean().item(), 'costs_std_max': costs_std_discrete.max().item()}
            if self.debug_mode_return_full_cost_std:
                return torch.cat((torch.tensor(0, device=self.d, dtype=self.dtype).view(1), costs.std(dim=1))).repeat_interleave(self.discrete_interval).cpu()

        if self.sampling_policy == 'discrete_monitoring':
            actions = self.U[costs_std_discrete < self.observing_var_threshold]
            if actions.shape[0] == 0:
                actions = self.U[:self.u_per_command]
                costs_std_discrete = costs_std_discrete[:self.u_per_command]
            else:
                costs_std_discrete = costs_std_discrete[costs_std_discrete < self.observing_var_threshold]
        elif self.sampling_policy == 'discrete_planning' or self.sampling_policy == 'continuous_planning':
            if self.fixed_continuous_planning_steps is None:
                if not self.debug_mode_cp_return_continuous_reward_unc:
                    actions = self.U[:self.observing_fixed_frequency] 
                    costs_std_discrete = costs_std_discrete[:self.observing_fixed_frequency]
                else:
                    actions = self.U[:self.observing_fixed_frequency] 
                    costs_std_continuous = costs_std_continuous[:self.observing_fixed_frequency * self.continuous_time_interval]
                    costs_std_discrete = torch.tensor(0, device=self.d, dtype=self.dtype).view(1)
            else:
                actions = self.U
                costs_std_discrete = costs_std_discrete
        elif self.sampling_policy == 'active_observing_control':
            actions = self.U
            actions = actions.repeat_interleave(self.discrete_interval, dim=0)
            slice_to_take_holder = torch.zeros((actions.shape[0])).bool()
            slice_to_take_holder[:select_actions_up_to] = True
            actions = actions[slice_to_take_holder]
            if actions.shape[0] <= (self.continuous_time_interval-1):
                self.previous_step = int(np.ceil(actions.shape[0] / self.discrete_interval))
                actions = self.U.repeat_interleave(self.discrete_interval,dim=0)
                actions = actions[:self.continuous_time_interval]
            else:
                self.previous_step = int(actions.shape[0] / self.discrete_interval)
            assert not torch.isnan(actions).any(), "Nan detected in actions"
            costs_std_continuous = torch.ones_like(actions).to(self.d)
            return actions * self.u_scale, costs_std_continuous, stats
        else:
            raise NotImplementedError(f'sampling_policy: {self.sampling_policy} not recognized')
        self.previous_step = actions.shape[0]
        assert not torch.isnan(actions).any(), "Nan detected in actions"
        if self.discrete_planning:
            actions = actions.repeat_interleave(self.discrete_interval,dim=0)
            costs_std_discrete = costs_std_discrete.repeat_interleave(self.discrete_interval,dim=0)
            if self.sampling_policy == 'continuous_planning':
                if self.fixed_continuous_planning_steps is None:
                    actions = actions[:self.continuous_time_interval]
                    if not self.debug_mode_cp_return_continuous_reward_unc:
                        costs_std_discrete = costs_std_discrete[:self.continuous_time_interval]
                    else:
                        costs_std_discrete = costs_std_continuous
                    self.previous_step = int(np.ceil(actions.shape[0] / self.discrete_interval))
                else:
                    actions = actions[:self.fixed_continuous_planning_steps]
                    costs_std_discrete = costs_std_discrete[:self.fixed_continuous_planning_steps]
                    self.previous_step = int(np.ceil(actions.shape[0] / self.discrete_interval))
        return actions * self.u_scale, costs_std_discrete, stats

def run_mppi(mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True):
    dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d)
    total_reward = 0
    for i in tqdm(range(iter)):
        state = env.state
        command_start = time.perf_counter()
        action = mppi.command(state)
        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action.cpu().numpy())
        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
        dataset[di, mppi.nx:] = action
    return total_reward, dataset
