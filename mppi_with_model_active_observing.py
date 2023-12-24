import time
from functools import partial

import imageio
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
from tqdm import tqdm

from config import dotdict
from overlay import create_env, plot_telem, setup_logger, start_virtual_display, step_env
from planners.mppi_active_observing import MPPIActiveObserving

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import deque
from copy import deepcopy

from torch.multiprocessing import get_logger

logger = get_logger()


def mppi_with_model_evaluate_single_step_active_observing(
    model_name,
    env_name,
    sampling_policy,
    roll_outs=1000,
    time_steps=30,
    lambda_=1.0,
    threshold=None,
    fixed_continuous_planning_observations=None,
    sigma=1.0,
    dt=0.05,
    model_seed=11,
    save_video=False,
    state_constraint=False,
    change_goal=False,
    encode_obs_time=False,
    model=None,
    uniq=None,
    observing_cost=0,
    config={},
    planner="mppi_active_observing",
    plot_seed=0,
    intermediate_run=False,
    seed=None,
):
    MODELS = ["pe", "pe-discrete", "oracle", "random"]
    SAMPLING_POLICIES = [
        "discrete_planning",
        "discrete_monitoring",
        "continuous_planning",
        "active_observing_control",
        "random",
    ]
    if env_name == "oderl-cancer":
        dt = 0.4
        observing_cost = observing_cost * 5
        config.update({"dt": dt})
        config.update({"discrete_interval": int(dt / config.dt_simulation)})
        config.update({"observing_cost": observing_cost})
    assert sampling_policy in SAMPLING_POLICIES
    assert model_name in MODELS
    from config import load_observing_var_thresholds

    var_thresholds_d = load_observing_var_thresholds()
    if sampling_policy == "active_observing_control" or sampling_policy == "continuous_planning":
        if threshold is not None:
            observing_var_threshold = threshold
        else:
            observing_var_threshold = var_thresholds_d[dt][env_name]["continuous"]
    else:  # Discrete
        observing_var_threshold = var_thresholds_d[dt][env_name]["discrete"]
    env = create_env(env_name, dt=config.dt_simulation, friction=config.friction)
    if env_name == "oderl-cancer":
        limit_actions_to_only_positive = True
    else:
        limit_actions_to_only_positive = False
    ACTION_LOW = env.action_space.low[0]
    ACTION_HIGH = env.action_space.high[0]

    nx = env.get_obs().shape[0]
    nu = env.action_space.shape[0]

    dtype = torch.float32
    gamma = sigma**2
    off_diagonal = 0.5 * gamma
    mppi_noise_sigma = torch.ones((nu, nu), device=device, dtype=dtype) * off_diagonal + torch.eye(
        nu, device=device, dtype=dtype
    ) * (gamma - off_diagonal)
    mppi_lambda_ = 1.0
    if not (model_name == "oracle" or model_name == "random"):
        if model is None:
            from train_utils import train_model

            model, results = train_model(
                model_name,
                env_name,
                config=config,
                wandb=None,
                model_seed=config.model_seed,
                retrain=False,
                start_from_checkpoint=True,
                force_retrain=False,
                print_settings=False,
                evaluate_model_when_trained=False,
            )

        def dynamics(
            state, perturbed_action, ts_pred, encode_obs_time=encode_obs_time, model_name=model_name, return_var=True
        ):
            if encode_obs_time and model_name == "nl":
                perturbed_action = torch.cat(
                    (perturbed_action, torch.ones(1, device=device))
                    .view(1, 1, 1)
                    .repeat(perturbed_action.shape[0], 1, 1)
                )
                assert False, "untested"
            state_diff_pred_mu, state_diff_pred_var = model(state, perturbed_action, ts_pred)
            state_out_mu = state + state_diff_pred_mu
            return state_out_mu, state_diff_pred_var

    elif model_name == "random":

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

        oracle_var_monte_carlo_samples = 100

        if config.oracle_var_type == "fixed_oracle_var":

            def dynamics(*args, **kwargs):
                state_mu = dynamics_oracle(*args, **kwargs)
                return state_mu, torch.ones_like(state_mu) * oracle_sigma

        elif config.oracle_var_type == "state_oracle_var":

            def dynamics(*args, **kwargs):
                if not kwargs["return_var"]:
                    state_mu = dynamics_oracle(*args, **kwargs)
                    return state_mu, None
                else:
                    state, perturbed_action, ts_pred = args
                    K, nu = perturbed_action.shape[0], perturbed_action.shape[1]
                    K, nx = state.shape[0], state.shape[1]
                    # state = torch.rand(1000,5).to(device)
                    state_samples = state.view(K, 1, nx).repeat(1, oracle_var_monte_carlo_samples, 1) + torch.normal(
                        0, oracle_sigma, size=(K, oracle_var_monte_carlo_samples, nx)
                    ).to(device)
                    state_estimates = dynamics_oracle(
                        state_samples.view(-1, nx),
                        perturbed_action.repeat_interleave(oracle_var_monte_carlo_samples, dim=0),
                        ts_pred.repeat(1, oracle_var_monte_carlo_samples).view(-1, 1),
                        **kwargs,
                    )
                    state_mus = state_estimates.view(K, oracle_var_monte_carlo_samples, nx).mean(dim=1)
                    state_vars = state_estimates.view(K, oracle_var_monte_carlo_samples, nx).var(dim=1)
                    return state_mus, state_vars

        elif config.oracle_var_type == "action_oracle_var":

            def dynamics(*args, **kwargs):
                if not kwargs["return_var"]:
                    state_mu = dynamics_oracle(*args, **kwargs)
                    return state_mu, None
                else:
                    state, perturbed_action, ts_pred = args
                    K, nu = perturbed_action.shape[0], perturbed_action.shape[1]
                    perturbed_action_samples = perturbed_action.view(K, 1, nu).repeat(
                        1, oracle_var_monte_carlo_samples, 1
                    ) + torch.normal(0, oracle_sigma, size=(K, oracle_var_monte_carlo_samples, nu)).to(device)
                    state_estimates = dynamics_oracle(
                        state.repeat_interleave(oracle_var_monte_carlo_samples, dim=0),
                        perturbed_action_samples.view(-1, nu),
                        ts_pred.repeat(1, oracle_var_monte_carlo_samples).view(-1, 1),
                        **kwargs,
                    )
                    state_estimates = state_estimates.view(K, oracle_var_monte_carlo_samples, -1)
                    state_mus = state_estimates.mean(dim=1)
                    state_vars = state_estimates.var(dim=1)
                return state_mus, state_vars

        else:
            raise NotImplementedError

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

    videos_folder = "./logs/new_videos"
    from pathlib import Path

    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    filename = f"{videos_folder}/{env_name}_{model_name}_{uniq}.mp4"
    fps = int(1 / config.dt_simulation)
    env.reset()
    state = env.get_obs()
    if not config.multi_process_results:
        logger.info(f"[Start State] {state}")

    global change_goal_flipped
    change_goal_flipped = False
    print_out_costs_var = False
    timelen = 5  # 50 for cancer # 5 for oderl envs
    if change_goal:
        timelen = timelen * 2.0
    iter_ = timelen / config.dt_simulation
    change_goal_flipped_iter_ = iter_ / 2.0
    if fixed_continuous_planning_observations is not None:
        fixed_continuous_planning_steps = int(iter_ / fixed_continuous_planning_observations) + 1
    else:
        fixed_continuous_planning_steps = None

    if env_name == "oderl-pendulum":

        def cost_var_from_state_var(state):
            return state.sum()

    elif env_name == "oderl-cartpole":

        def cost_var_from_state_var(state):
            # return state[...,[0,2]].sum()
            return state.sum()

    elif env_name == "oderl-acrobot":

        def cost_var_from_state_var(state):
            return state.sum()

    elif env_name == "oderl-cancer":

        def cost_var_from_state_var(state):
            return state.sum()

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
        observing_cost=observing_cost,
        sampling_policy=sampling_policy,
        continuous_time_threshold=config.continuous_time_threshold,
        observing_var_threshold=observing_var_threshold,
        observing_fixed_frequency=config.observing_fixed_frequency,
        dt_simulation=config.dt_simulation,
        dt=dt,
        cost_var_from_state_var=cost_var_from_state_var,
        discrete_planning=config.discrete_planning,
        discrete_interval=config.discrete_interval,
        limit_actions_to_only_positive=limit_actions_to_only_positive,
        fixed_continuous_planning_steps=fixed_continuous_planning_steps,
    )
    mppi_gym.reset()

    if save_video:
        start_virtual_display()

    def loop():
        it = 0
        state_reward = 0
        observations_taken = 0
        start_time = time.perf_counter()
        episode_elapsed_time = 0
        actions_to_execute = []
        observed_times = []
        costs_std_median = []
        costs_std_all = []
        s = []
        a = []
        r = []
        ri = []
        costs_std_stats_l = []
        while it < iter_:
            if change_goal_flipped_iter_ < it:
                change_goal_flipped = True
            if not actions_to_execute:
                state = env.get_obs()
                observations_taken += 1
                observed_times.append(it)
                command_start = time.perf_counter()
                t0 = time.perf_counter()
                if model_name == "random" or sampling_policy == "random":
                    actions = torch.from_numpy(env.action_space.sample()).view(1, -1)
                    actions = actions.repeat_interleave(config.discrete_interval, dim=0)
                    if env_name == "oderl-cartpole":
                        actions = torch.zeros_like(actions)
                    costs_std = torch.zeros(actions.shape[0], device=actions.device)
                    costs_std_all.append(costs_std)
                else:
                    # MPC command
                    actions, costs_std, costs_std_stats = mppi_gym.command(state)
                    assert actions.shape[0] == costs_std.shape[0], "Shapes must match"
                    costs_std_all.append(costs_std)
                    costs_std_stats_l.append(costs_std_stats)
                actions_to_execute = deque(list(actions))
                episode_elapsed_time += time.perf_counter() - t0
                elapsed = time.perf_counter() - command_start
            action = actions_to_execute.popleft()
            s.append(state)
            a.append(action)
            # elapsed = time.perf_counter() - command_start
            state, reward, done, _ = step_env(env, action.detach().cpu().numpy(), obs_noise=config.observation_noise)
            reward = reward.detach().cpu().item()
            state_reward += reward
            r.append(state_reward)
            ri.append(reward)
            if not config.multi_process_results:
                if print_out_costs_var:
                    logger.info(
                        f"[{env_name}\t{model_name}\t|time_steps={time_steps}__dt_sim={config.dt_simulation}] action taken: {action.detach().cpu().numpy()} cost received: {-reward} | state {state.flatten()} time taken: {elapsed}s | {int(it/iter_*100)}% Complete \t | iter={it} \t| observed_times_diff={np.diff(np.array(observed_times))} \t| costs_var_sum={np.array(costs_std_median)}"
                    )
                else:
                    logger.info(
                        f"[{env_name}\t{model_name}\t|time_steps={time_steps}__dt_sim={config.dt_simulation}] action taken: {action.detach().cpu().numpy()} cost received: {-reward} | state {state.flatten()} time taken: {elapsed}s | {int(it/iter_*100)}% Complete \t | iter={it} \t| observed_times_diff={np.diff(np.array(observed_times))}"
                    )
            if save_video:
                video.append_data(env.render(mode="rgb_array", last_act=action.detach().cpu().numpy()))
            it += 1

        observations_taken = observations_taken
        observation_reward = -observations_taken * observing_cost
        ddict = {
            "model_name": model_name,
            "env_name": env_name,
            "roll_outs": roll_outs,
            "time_steps": time_steps,
            "uniq": uniq,
            "episode_elapsed_time": episode_elapsed_time,
            "episode_elapsed_time_per_it": episode_elapsed_time / it,
            "dt_sim": config.dt_simulation,
            "dt_plan": dt,
            "planner": "mpc",
            "total_reward": (np.array(ri).sum() + observation_reward) / iter_,
            "state_reward": np.array(ri).mean(),
            "state_reward_std": np.array(ri).std(),
            "observation_reward": observation_reward / iter_,
            "observations_taken": observations_taken,
            "sampling_policy": sampling_policy,
            "observing_var_threshold": observing_var_threshold,
            "observing_cost": config.observing_cost,
            "observed_times": observed_times,
            "observed_times_diff": np.diff(np.array(observed_times)).tolist(),
            "costs_std_median": np.array(costs_std_median).tolist(),
            "observation_noise": config.observation_noise,
            "fixed_continuous_planning_observations": fixed_continuous_planning_observations,
        }
        if config.plot_telem:
            a = torch.stack(a)
            a = a.detach().cpu().numpy()
            r = np.array(r)
            s = np.stack(s)
            ri = np.array(ri)
            df = pd.DataFrame(costs_std_stats_l)
            cost_std_plot = torch.cat(costs_std_all)[: s.shape[0]].detach().cpu().numpy()
            assert cost_std_plot.shape[0] == s.shape[0], f"Shape cost_std_plot: {cost_std_plot.shape}"
            ddict.update(
                {
                    "s": s,
                    "a": a,
                    "r": r,
                    "cost_std_plot": cost_std_plot,
                    "ri": ri,
                    "plot_seed": plot_seed,
                    "costs_std_stats": df.to_json().replace("{", "<").replace("}", ">"),
                }
            )
            # print(f"THRESHOLD: {df.mean()['costs_std_median']}")
        if not config.multi_process_results:
            if save_video:
                logger.info(f"[{env_name}\t{model_name}\t][Video] Watch video at : {filename}")
            if intermediate_run:
                logger.info(f"[{env_name}\t{model_name}\t][Intermediate Result] {str(ddict)}")
            else:
                logger.info(f"[{env_name}\t{model_name}\t][Result] {str(ddict)}")
        return ddict

    with torch.no_grad():
        if save_video:
            with imageio.get_writer(filename, fps=fps) as video:
                result = loop()
        else:
            result = loop()
    if config.plot_telem:
        telem_file_path = plot_telem(result)
        result.update({"telem_file_path": telem_file_path})
    return result


def seed_wrapper_mppi_with_model_evaluate_single_step_active_observing(
    run_args,
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
    observing_cost=0,
    config={},
    planner="mppi",
    intermediate_run=False,
):
    (env_name, model_name, sampling_policy, seed, threshold, fixed_continuous_planning_observations) = run_args
    from config import seed_all

    seed_all(seed)
    config = dotdict(deepcopy(dict(config)))
    result = mppi_with_model_evaluate_single_step_active_observing(
        model_name=model_name,
        env_name=env_name,
        threshold=threshold,
        fixed_continuous_planning_observations=fixed_continuous_planning_observations,
        sampling_policy=sampling_policy,
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
        observing_cost=observing_cost,
        config=config,
        planner=planner,
        plot_seed=seed,
        intermediate_run=intermediate_run,
    )
    result.update({"seed": seed})
    return result


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    import wandb

    from config import get_config

    defaults = get_config()
    defaults["save_video"] = False
    defaults["seed_start"] = 0
    defaults["seed_runs"] = 1000  # Final results 1000 random seeds
    defaults["collect_expert_cores_per_env_sampler"] = 4

    debug_main = False
    defaults["plot_telem"] = debug_main
    defaults["multi_process_results"] = not debug_main
    planner = "mppi_active_observing"  # 'mppi'
    wandb.init(config=defaults, project=defaults["wandb_project"], mode="disabled")
    config = wandb.config
    logger = setup_logger(__file__)
    from tqdm import tqdm

    if not debug_main:
        pool_outer = multiprocessing.Pool(config.collect_expert_cores_per_env_sampler)
    args_for_runs = []
    t0 = time.perf_counter()
    threshold = None  # Automatically set by tuning
    fixed_continuous_planning_observations = None
    for env_name in [
        "oderl-cancer",
        "oderl-acrobot",
        "oderl-pendulum",
        "oderl-cartpole",
    ]:
        for sampling_policy, model_name in [
            ("active_observing_control", "pe"),
            ("continuous_planning", "pe"),
            ("discrete_planning", "pe-discrete"),
            ("discrete_monitoring", "pe-discrete"),
            ("random", "random"),
        ]:
            for seed in range(config.seed_start, config.seed_runs + config.seed_start):
                args_for_runs.append(
                    (env_name, model_name, sampling_policy, seed, threshold, fixed_continuous_planning_observations)
                )
    multi_seed_wrapper_mppi_with_model_evaluate_single_step = partial(
        seed_wrapper_mppi_with_model_evaluate_single_step_active_observing,
        roll_outs=config.mppi_roll_outs,
        time_steps=config.mppi_time_steps,
        lambda_=config.mppi_lambda,
        sigma=config.mppi_sigma,
        dt=config.dt,
        uniq=0,
        observing_cost=config.observing_cost,
        encode_obs_time=config.encode_obs_time,
        config=dict(config),
        planner=planner,
        save_video=config.save_video,
    )
    results = []
    if debug_main:
        for args_for_run in args_for_runs:
            result = multi_seed_wrapper_mppi_with_model_evaluate_single_step(args_for_run)
            printable_result = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
            logger.info(f"[Policy evaluation complete] {printable_result}")
            logger.info(
                f"[Policy short form][{printable_result['env_name']}|{printable_result['sampling_policy']}|{printable_result['model_name']}|{printable_result['dt_plan']}] total_reward:{printable_result['total_reward']}\t| state_reward:{printable_result['state_reward']}\t| observations_taken:{printable_result['observations_taken']}\t| observing_var_threshold:{printable_result['observing_var_threshold']}\t|"
            )
            if config.plot_telem:
                logger.info(f"[Policy telem path] {result['telem_file_path']}")
            results.append(result)
    else:
        for i, result in tqdm(
            enumerate(
                pool_outer.imap_unordered(multi_seed_wrapper_mppi_with_model_evaluate_single_step, args_for_runs)
            ),
            total=len(args_for_runs),
            smoothing=0,
        ):
            printable_result = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
            logger.info(f"[Policy evaluation complete] {printable_result}")
            logger.info(
                f"[Policy short form][{printable_result['env_name']}|{printable_result['sampling_policy']}|{printable_result['model_name']}|{printable_result['dt_plan']}] total_reward:{printable_result['total_reward']}\t| state_reward:{printable_result['state_reward']}\t| observations_taken:{printable_result['observations_taken']}\t| observing_var_threshold:{printable_result['observing_var_threshold']}\t|"
            )
            if config.plot_telem:
                logger.info(f"[Policy telem path] {result['telem_file_path']}")
            results.append(result)
    time_taken = time.perf_counter() - t0
    logger.info(f"Time taken for all runs: {time_taken}s\t| {time_taken/60.0} minutes")
    df_results = pd.DataFrame(results)
    from process_results.plot_util import generate_main_results_table, normalize_means

    df = normalize_means(df_results)
    df_out, table_str = generate_main_results_table(df)
    print("")
    print(table_str)
    print("fin.")
