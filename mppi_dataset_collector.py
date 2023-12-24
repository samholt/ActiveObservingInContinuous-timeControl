import logging
import os
import time
from functools import partial

import imageio
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from tqdm import tqdm

from config import dotdict
from overlay import create_env, setup_logger, start_virtual_display, step_env
from planners.mppi import MPPI
from planners.mppi_active_observing import MPPIActiveObserving

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()


def inner_mppi_with_model_collect_data(
    seed,
    model_name,
    env_name,
    roll_outs=1000,
    time_steps=30,
    lambda_=1.0,
    sigma=1.0,
    dt=0.05,
    model_seed=11,
    save_video=False,
    state_constraint=False,
    change_goal=False,
    encode_obs_time=False,
    model=None,
    uniq=None,
    log_debug=False,
    episodes_per_sampler_task=10,
    config={},
    iter_=200,
    change_goal_flipped_iter_=False,
    ts_grid="exp",
    intermediate_run=False,
):
    config = dotdict(config)
    env = create_env(env_name, dt=dt, ts_grid=ts_grid, friction=config.friction)
    ACTION_LOW = env.action_space.low[0]
    ACTION_HIGH = env.action_space.high[0]
    if env_name == "oderl-cancer":
        limit_actions_to_only_positive = True
    else:
        limit_actions_to_only_positive = False

    nx = env.get_obs().shape[0]
    nu = env.action_space.shape[0]

    dtype = torch.float32
    gamma = sigma**2
    off_diagonal = 0.5 * gamma
    mppi_noise_sigma = torch.ones((nu, nu), device=device, dtype=dtype) * off_diagonal + torch.eye(
        nu, device=device, dtype=dtype
    ) * (gamma - off_diagonal)
    logger.info(mppi_noise_sigma)
    mppi_lambda_ = 1.0

    random_action_noise = config.collect_expert_random_action_noise

    if model_name == "random":

        def dynamics(state, perturbed_action):
            pass

    elif model_name == "oracle":
        oracle_sigma = config.observation_noise
        if env_name == "oderl-pendulum":
            from oracle import pendulum_dynamics_dt

            dynamics_oracle = pendulum_dynamics_dt
        elif env_name == "oderl-cartpole":
            from oracle import cartpole_dynamics_dt

            dynamics_oracle = cartpole_dynamics_dt
        elif env_name == "oderl-acrobot":
            from oracle import acrobot_dynamics_dt

            dynamics_oracle = acrobot_dynamics_dt
        elif env_name == "oderl-cancer":
            from oracle import cancer_dynamics_dt

            dynamics_oracle = cancer_dynamics_dt

        def dynamics(*args, **kwargs):
            state_mu = dynamics_oracle(*args, **kwargs)
            return state_mu, torch.ones_like(state_mu) * oracle_sigma

        dynamics = partial(dynamics, friction=config.friction)

    def running_cost(state, action):
        if state_constraint:
            reward = env.diff_obs_reward_(
                state, exp_reward=False, state_constraint=state_constraint
            ) + env.diff_ac_reward_(action)
        elif change_goal:
            global change_goal_flipped
            reward = env.diff_obs_reward_(
                state, exp_reward=False, change_goal=change_goal, change_goal_flipped=change_goal_flipped
            ) + env.diff_ac_reward_(action)
        else:
            reward = env.diff_obs_reward_(state, exp_reward=False) + env.diff_ac_reward_(action)
        cost = -reward
        return cost

    if config.planner == "mppi":
        mppi_gym = MPPI(
            dynamics,
            running_cost,
            nx,
            mppi_noise_sigma,
            num_samples=roll_outs,
            horizon=time_steps,
            device=device,
            lambda_=mppi_lambda_,
            u_min=torch.tensor(ACTION_LOW),
            u_max=torch.tensor(ACTION_HIGH),
            u_scale=ACTION_HIGH,
        )
    elif config.planner == "mppi_active_observing":
        mppi_gym = MPPIActiveObserving(
            dynamics,
            running_cost,
            nx,
            mppi_noise_sigma,
            num_samples=roll_outs,
            horizon=time_steps,
            device=device,
            lambda_=mppi_lambda_,
            u_min=torch.tensor(ACTION_LOW),
            u_max=torch.tensor(ACTION_HIGH),
            u_scale=ACTION_HIGH,
            observing_cost=config.observing_cost,
            sampling_policy=config.sampling_policy,
            observing_var_threshold=config.observing_var_threshold,
            limit_actions_to_only_positive=limit_actions_to_only_positive,
            dt=dt,
        )

    if save_video:
        start_virtual_display()

    videos_folder = "./logs/new_videos"
    from pathlib import Path

    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    filename = f"{videos_folder}/{env_name}_{model_name}_{uniq}.mp4"
    fps = int(1 / dt)

    def loop():
        s0 = []
        a0 = []
        sn = []
        ts = []
        ACTION_LOW = env.action_space.low[0]
        ACTION_HIGH = env.action_space.high[0]
        it = 0
        total_reward = 0
        env.reset()
        start_time = time.perf_counter()
        mppi_gym.reset()
        while it < iter_:
            if change_goal_flipped_iter_ < it:
                change_goal_flipped = True
            state = env.get_obs()
            s0.append(state)
            command_start = time.perf_counter()
            if model_name != "random":
                action, costs_std = mppi_gym.command(state)
                if random_action_noise is not None:
                    action += (
                        (torch.rand(nu, device=device) - 0.5) * 2.0 * env.action_space.high[0]
                    ) * random_action_noise
                    action = action.clip(min=ACTION_LOW, max=ACTION_HIGH)
                    action = action.view(nu)
            else:
                action = torch.from_numpy(env.action_space.sample())
            elapsed = time.perf_counter() - command_start
            state, reward, done, tsn = step_env(env, action.detach().cpu().numpy(), obs_noise=config.observation_noise)
            sn.append(state)
            a0.append(action)
            ts.append(tsn)
            total_reward += reward
            if log_debug:
                logger.info(
                    f"action taken: {action.detach().cpu().numpy()} cost received: {-reward} | state {state.flatten()} ts {tsn.detach().cpu().numpy()} | time taken: {elapsed}s | {int(it/iter_*100)}% Complete \t | iter={it}"
                )
            if save_video:
                video.append_data(env.render(mode="rgb_array", last_act=action.detach().cpu().numpy()))
            it += 1
        total_reward = total_reward.detach().cpu().item()
        ddict = {
            "model_name": model_name,
            "env_name": env_name,
            "roll_outs": roll_outs,
            "time_steps": time_steps,
            "uniq": uniq,
            "episode_elapsed_time": time.perf_counter() - start_time,
            "dt": dt,
            "planner": "mpc",
            "total_reward": total_reward,
        }
        if save_video:
            logger.info(f"[Video] Watch video at : {filename}")
        if intermediate_run:
            logger.info(f"[Intermediate Result] {str(ddict)}")
        else:
            logger.info(f"[Result] {str(ddict)}")
        s0 = torch.from_numpy(np.stack(s0))
        sn = torch.from_numpy(np.stack(sn))
        a0 = torch.stack(a0).cpu()
        ts = torch.stack(ts).cpu()
        return ddict, (s0, a0, sn, ts)

    episodes = []
    for j in range(episodes_per_sampler_task):
        with torch.no_grad():
            if save_video:
                with imageio.get_writer(filename, fps=fps) as video:
                    try:
                        result, episode_buffer = loop()
                        episodes.append(episode_buffer)
                    except Exception as e:
                        logger.info(f"[Error] Error collecting episode : {e}")
            else:
                try:
                    result, episode_buffer = loop()
                    episodes.append(episode_buffer)
                except Exception as e:
                    logger.info(f"[Error] Error collecting episode : {e}")
    return episodes


def mppi_with_model_collect_data(
    model_name,
    env_name,
    roll_outs=1000,
    time_steps=30,
    lambda_=1.0,
    sigma=1.0,
    dt=0.05,
    model_seed=11,
    save_video=False,
    state_constraint=False,
    change_goal=False,
    encode_obs_time=False,
    model=None,
    uniq=None,
    log_debug=False,
    collect_samples=1e6,
    config_in={},
    debug_main=False,
    ts_grid="exp",
    intermediate_run=False,
):
    config = dotdict(dict(config_in))

    file_name = f"replay_buffer_env-name-{env_name}_model-name-{model_name}_encode-obs-time-{encode_obs_time}_ts-grid-{ts_grid}_random-action-noise-{config.collect_expert_random_action_noise}_observation-noise-{config.observation_noise}_friction-{config.friction}.pt"
    if not config.collect_expert_force_generate_new_data:
        final_data = torch.load(f"./offlinedata/{file_name}")
        return final_data

    global change_goal_flipped
    change_goal_flipped = False
    timelen = 10  # seconds
    if change_goal:
        timelen = timelen * 2.0
    iter_ = timelen / dt
    change_goal_flipped_iter_ = iter_ / 2.0

    multi_inner_mppi_with_model_collect_data = partial(
        inner_mppi_with_model_collect_data,
        model_name=model_name,
        env_name=env_name,
        roll_outs=roll_outs,
        time_steps=time_steps,
        lambda_=lambda_,
        sigma=sigma,
        dt=dt,
        model_seed=model_seed,
        save_video=save_video,
        state_constraint=state_constraint,
        change_goal=change_goal,
        encode_obs_time=encode_obs_time,
        model=model,
        uniq=uniq,
        log_debug=log_debug,
        episodes_per_sampler_task=config.collect_expert_episodes_per_sampler_task,
        config=dict(config),
        ts_grid=ts_grid,
        iter_=iter_,
        change_goal_flipped_iter_=change_goal_flipped_iter_,
        intermediate_run=intermediate_run,
    )
    total_episodes_needed = int(collect_samples / iter_)
    task_inputs = [
        run_seed for run_seed in range(int(total_episodes_needed / config.collect_expert_episodes_per_sampler_task))
    ]
    episodes = []
    if not debug_main:
        pool_outer = multiprocessing.Pool(config.collect_expert_cores_per_env_sampler)
        for i, result in tqdm(
            enumerate(pool_outer.imap_unordered(multi_inner_mppi_with_model_collect_data, task_inputs)),
            total=len(task_inputs),
            smoothing=0,
        ):
            episodes.extend(result)
    else:
        for i, task in tqdm(enumerate(task_inputs), total=len(task_inputs)):
            result = multi_inner_mppi_with_model_collect_data(task)
            episodes.extend(result)

    s0 = []
    sn = []
    a0 = []
    ts = []
    for episode in episodes:
        (es0, ea0, esn, ets) = episode
        s0.append(es0)
        sn.append(esn)
        a0.append(ea0)
        ts.append(ets)
    s0 = torch.cat(s0, dim=0)
    sn = torch.cat(sn, dim=0)
    a0 = torch.cat(a0, dim=0)
    ts = torch.cat(ts, dim=0).view(-1, 1)
    final_data = (s0, a0, sn, ts)
    if not os.path.exists("./offlinedata/"):
        os.makedirs("./offlinedata/")
    torch.save(final_data, f"./offlinedata/{file_name}")
    pool_outer.close()
    return final_data


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    from config import get_config, seed_all

    defaults = get_config()
    debug_collector = False
    defaults["save_video"] = False
    defaults["mppi_time_steps"] = 40
    defaults["collect_expert_force_generate_new_data"] = True
    defaults["collect_expert_cores_per_env_sampler"] = 6
    defaults["sampling_policy"] = "discrete_planning"
    defaults["observing_fixed_frequency"] = 1
    defaults["planner"] = "mppi_active_observing"  # 'mppi'
    defaults["dt"] = 0.05
    config = dotdict(defaults)
    seed_all(0)

    logger = setup_logger(__file__)
    for env_name in ["oderl-cartpole", "oderl-acrobot", "oderl-pendulum"]:
        logger.info(f"[Collecting data expert data] env_name={env_name}")
        results = mppi_with_model_collect_data(
            model_name="oracle",
            env_name=env_name,
            roll_outs=config.mppi_roll_outs,
            time_steps=config.mppi_time_steps,
            lambda_=config.mppi_lambda,
            sigma=config.mppi_sigma,
            dt=config.dt,
            collect_samples=config.collect_expert_samples,
            uniq=None,
            debug_main=debug_collector,
            encode_obs_time=config.encode_obs_time,
            ts_grid=config.ts_grid,
            config_in=config,
            log_debug=debug_collector,
            save_video=config.save_video,
        )
    logger.info("Fin.")
