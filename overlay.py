import os
import gym
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
from envs.oderl import envs
import numpy as np
TORCH_PRECISION = torch.float32

def step_env(env, action, obs_noise):
        at = torch.from_numpy(action).to(device)
        def g(state, t): return at
        returns = env.integrate_system(2, g, s0=torch.tensor(env.state).to(device), return_states=True)
        state = returns[-1][-1]
        reward = returns[2][-1]
        tsn = returns[-2][-1,-1]
        env.set_state_(state.cpu().numpy())
        state_out = env.get_obs()
        state_out = torch.from_numpy(state_out).to(device)
        state_out += torch.randn_like(state_out) * obs_noise
        env.time_step += 1
        done = True if env.time_step >= env.n_steps else False
        state_out = state_out.cpu().numpy()
        return state_out, reward, done, tsn

def start_virtual_display():
    import pyvirtualdisplay
    return pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

def create_oderl_env(env_name, dt=0.05, ts_grid='fixed', noise=0.0, friction=False, device=device):
    ################## environment and dataset ##################
    # dt      = 0.05 		# mean time difference between observations
    # noise   = 0.0 		# observation noise std
    # ts_grid = 'fixed' 	# the distribution for the observation time differences: ['fixed','uniform','exp']
    # ENV_CLS = envs.CTCartpole # [CTPendulum, CTCartpole, CTAcrobot]
    if env_name == "oderl-pendulum":
        ENV_CLS = envs.CTPendulum # [CTPendulum, CTCartpole, CTAcrobot]
    elif env_name == "oderl-cartpole":
        ENV_CLS = envs.CTCartpole # [CTPendulum, CTCartpole, CTAcrobot]
    elif env_name == "oderl-acrobot":
        ENV_CLS = envs.CTAcrobot # [CTPendulum, CTCartpole, CTAcrobot]
    elif env_name == "oderl-cancer":
        ENV_CLS = envs.CTCancer # [CTPendulum, CTCartpole, CTAcrobot]
    else:
        raise ValueError(f"Unknown enviroment: {env_name}")
    env = ENV_CLS(dt=dt, obs_trans=True, device=device, obs_noise=noise, ts_grid=ts_grid, solver='euler', friction=friction)
    return env

def create_env(env_name, dt=0.05, ts_grid='fixed', noise=0.0, friction=False, device=device):
    if 'oderl' in env_name:
        env = create_oderl_env(env_name, dt=dt, ts_grid=ts_grid, noise=noise, friction=friction, device=device)
    else:
        env = gym.make(env_name)
    return env

def setup_logger(file, log_folder='logs', return_path_to_log=False):
    import os, time, logging
    file_name = os.path.basename(os.path.realpath(file)).split('.py')[0]
    from pathlib import Path
    Path(f"./{log_folder}").mkdir(parents=True, exist_ok=True)
    path_run_name = '{}-{}'.format(
        file_name, time.strftime("%Y%m%d-%H%M%S"))
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(f"{log_folder}/{path_run_name}_log.txt"),
            logging.StreamHandler(),
        ],
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    logger.info(f'Starting: Log file at: {log_folder}/{path_run_name}_log.txt')
    if return_path_to_log:
        return logger, f'{log_folder}/{path_run_name}_log.txt'
    else:
        return logger

def compute_state_actions(rand, samples_per_dim, device_h, state_max, action_max, state_min, action_min):
    if state_min is None:
        state_min = -state_max
    if action_min is None:
        action_min = -action_max
    if rand:
        s0s_dist = ((torch.rand(samples_per_dim**state_max.shape[0], state_max.shape[0], dtype=TORCH_PRECISION, device=device_h) - 0.5) * 2.0) # [-1,1]
        s0_scale = (state_max - state_min) / 2.0
        s0_bias = (state_max + state_min) / 2.0
        s0s = s0s_dist * s0_scale + s0_bias
        actions_dist = ((torch.rand(samples_per_dim, action_max.shape[0], dtype=TORCH_PRECISION, device=device_h) - 0.5) * 2.0)
        actions_scale = (action_max - action_min) / 2.0
        actions_bias = (action_max + action_min) / 2.0
        actions = actions_dist * actions_scale + actions_bias
        actions = actions.view(-1,action_max.shape[0])
    else:
        if state_max.shape[0] == 4:
            x, y, z, k = torch.meshgrid(torch.linspace(state_min[0], state_max[0],samples_per_dim, device=device_h),
                                        torch.linspace(state_min[1], state_max[1],samples_per_dim, device=device_h),
                                        torch.linspace(state_min[2], state_max[2],samples_per_dim, device=device_h),
                                        torch.linspace(state_min[3], state_max[3],samples_per_dim, device=device_h),
                                        indexing='ij')
            all_t = torch.cat((x.unsqueeze(-1),
                                y.unsqueeze(-1),
                                z.unsqueeze(-1),
                                k.unsqueeze(-1)),-1)
            s0s = all_t.view(-1,4)
        elif state_max.shape[0] == 2:
            x, y = torch.meshgrid(torch.linspace(state_max[0], state_max[0],samples_per_dim, device=device_h),
                                  torch.linspace(state_max[1], state_max[1],samples_per_dim, device=device_h),
                                    indexing='ij')
            all_t = torch.cat((x.unsqueeze(-1),
                                y.unsqueeze(-1)),-1)
            s0s = all_t.view(-1,2)
        if action_max.shape[0] == 1:
            actions = torch.linspace(-action_max[0], action_max[0],samples_per_dim, device=device_h).view(-1,1)
        elif action_max.shape[0] == 2:
            a1, a2 = torch.meshgrid(torch.linspace(action_min[0], action_max[0],samples_per_dim, device=device_h),
                                    torch.linspace(action_min[1], action_max[1],samples_per_dim, device=device_h),
                                    indexing='ij')
            a_t = torch.cat((a1.unsqueeze(-1),
                            a2.unsqueeze(-1)),-1)
            actions = a_t.view(-1,2)
    return s0s, actions

def generate_irregular_data_time_multi(train_env_task,
                                        env,
                                        samples_per_dim=None,
                                        mode='grid',
                                        rand=False,
                                        encode_obs_time=False,
                                        reuse_state_actions_when_sampling_times=False,
                                        observation_noise=0):
    if samples_per_dim is None:
        if train_env_task == 'oderl-pendulum':
            samples_per_dim = 33
        if train_env_task == 'oderl-cancer':
            samples_per_dim = 20
        elif train_env_task == 'oderl-cartpole':
            samples_per_dim = 20
        elif train_env_task == 'oderl-acrobot':
            samples_per_dim = 15
    time_multiplier = 10
    ACTION_HIGH = env.action_space.high[0]
    # if train_env_task == 'oderl-cancer':
    #     ACTION_LOW = 0
    ACTION_LOW = env.action_space.low[0]
    nu = env.action_space.shape[0]
    s0_l, a0_l, sn_l, ts_l = [], [], [], []
    action_max = torch.tensor([ACTION_HIGH] * nu)
    device_h = 'cpu'
    state_min = None
    action_min = None
    if train_env_task == 'oderl-cartpole':
        state_max = torch.tensor([5.0, 20, torch.pi, 30]) # state_max = torch.tensor([7.0, 20, torch.pi, 30]) # state_max = torch.tensor([1.0, 1.0, torch.pi/8.0, 3.0])
    elif train_env_task == 'oderl-pendulum':
        state_max = torch.tensor([torch.pi, 5.0])
    elif train_env_task == 'oderl-acrobot':
        state_max = torch.tensor([torch.pi, torch.pi, 5.0, 5.0])
    elif train_env_task == 'oderl-cancer':
        state_max = torch.tensor([1160,10])
        state_min = torch.tensor([0,0])
        action_min = torch.tensor([0,0])
    if reuse_state_actions_when_sampling_times:
        s0s, actions = compute_state_actions(rand, samples_per_dim, device_h, state_max, action_max, state_min, action_min)
        for ti in range(int(samples_per_dim * time_multiplier)):
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h) # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1,1)
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)
    else:
        for ti in range(int(samples_per_dim * time_multiplier)):
            s0s, actions = compute_state_actions(rand, samples_per_dim, device_h, state_max, action_max, state_min, action_min)
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h) # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1,1)
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)

    s0 = torch.cat(s0_l, dim=0)
    a0 = torch.cat(a0_l, dim=0)
    sn = torch.cat(sn_l, dim=0)
    ts = torch.cat(ts_l, dim=0)

    # from oracle import pendulum_dynamics_dt
    # from oracle import acrobot_dynamics_dt
    # from oracle import cartpole_dynamics_dt
    # from oracle import cancer_dynamics_dt
    # sn = cancer_dynamics_dt(s0, a0, ts)
    # sn = pendulum_dynamics_dt(s0, a0, ts)
    # sn = acrobot_dynamics_dt(s0, a0, ts)

    # print(f'This should always be near zero: {((sn - cancer_dynamics_dt(s0, a0, ts))**2).mean()}') #  Can investigate to make zero
    # print(f'This should always be near zero: {((sn - cartpole_dynamics_dt(s0, a0, ts))**2).mean()}') #  Can investigate to make zero
    # print(f'This should always be near zero: {((sn - pendulum_dynamics_dt(s0, a0, ts))**2).mean()}')
    # print(f'This should always be near zero: {((sn - acrobot_dynamics_dt(s0, a0, ts))**2).mean()}')
    # if encode_obs_time:
    #     a0 = torch.cat((a0, torch.flip(torch.arange(action_buffer_size),(0,)).view(1,action_buffer_size,1).repeat(a0.shape[0],1,1)),dim=2)
    #     # a0 = torch.cat((a0.view(a0.shape[0],-1,nu),a),dim=1)

    # from oracle import cartpole_dynamics_dt_delay
    # print(f'This should always be near zero: {((sn - cartpole_dynamics_dt_delay(s0, a0, ts, delay=delay))**2).mean()}') #  Can investigate to make zero

    # from oracle import cartpole_dynamics_dt_delay
    # print(f'This should always be near zero: {((sn - cartpole_dynamics_dt_delay(s0, a0, ts, delay=delay))**2).mean()}')
    # from oracle import acrobot_dynamics_dt_delay
    # print(f'This should always be near zero: {((sn - acrobot_dynamics_dt_delay(s0, a0, ts, delay=delay))**2).mean()}')

    # from oracle import pendulum_dynamics_dt_delay
    # print(f'This should always be near zero: {((sn - pendulum_dynamics_dt_delay(s0, a0, ts, delay=delay))**2).mean()}')

    # s0 = s0.double()
    # a0 = a0.double()
    # sn = sn.double()
    # ts = ts.double()

    # Add observation noise
    sn += torch.randn_like(sn) * observation_noise
    return s0.detach(), a0.detach(), sn.detach(), ts.detach()

def load_expert_irregular_data_time_multi(train_env_task,
                                        encode_obs_time=True,
                                        config={}):
    from mppi_dataset_collector import mppi_with_model_collect_data
    final_data = mppi_with_model_collect_data(
                                'oracle', # 'nl', 'NN', 'oracle', 'random'
                                train_env_task,
                                roll_outs=config.mppi_roll_outs,
                                time_steps=config.mppi_time_steps,
                                lambda_=config.mppi_lambda,
                                sigma=config.mppi_sigma,
                                dt=config.dt,
                                save_video=False,
                                state_constraint=False,
                                change_goal=False,
                                encode_obs_time=encode_obs_time,
                                model=None,
                                uniq=None,
                                log_debug=False,
                                collect_samples=config.collect_expert_samples,
                                config_in=config,
                                ts_grid = config.collect_expert_ts_grid,
                                intermediate_run=False)
    (s0, a0, sn, ts) = final_data
    # from oracle import pendulum_dynamics_dt
    # from oracle import acrobot_dynamics_dt
    # from oracle import cartpole_dynamics_dt
    # sn = pendulum_dynamics_dt(s0, a0, ts)
    # sn = acrobot_dynamics_dt(s0, a0, ts)

    # print(f'This should always be near zero: {((sn - cartpole_dynamics_dt(s0, a0, ts))**2).mean()}') #  Can investigate to make zero
    # print(f'This should always be near zero: {((sn - pendulum_dynamics_dt(s0, a0, ts))**2).mean()}')
    # print(f'This should always be near zero: {((sn - acrobot_dynamics_dt(s0, a0, ts))**2).mean()}')


    # from oracle import cartpole_dynamics_dt_delay
    # print(f'This should always be near zero: {((sn - cartpole_dynamics_dt_delay(s0, a0, ts, delay=delay))**2).mean()}') #  Can investigate to make zero

    # from oracle import cartpole_dynamics_dt_delay
    # print(f'This should always be near zero: {((sn - cartpole_dynamics_dt_delay(s0, a0, ts, delay=delay))**2).mean()}')
    # from oracle import acrobot_dynamics_dt_delay
    # print(f'This should always be near zero: {((sn - acrobot_dynamics_dt_delay(s0, a0, ts, delay=delay))**2).mean()}')

    # from oracle import pendulum_dynamics_dt_delay
    # print(f'This should always be near zero: {((sn - pendulum_dynamics_dt_delay(s0, a0, ts, delay=delay))**2).mean()}')

    # s0 = s0.double()
    # a0 = a0.double()
    # sn = sn.double()
    # ts = ts.double()
    # assert not (s0[1:,:] - sn[:-1,:]).bool().any().item(), "Invariant failed, error in data generation"

    s0 = s0.float()
    a0 = a0.float()
    sn = sn.float()
    ts = ts.float()
    return s0.detach(), a0.detach(), sn.detach(), ts.detach()

################## Plotting ##################

def plot_telem(result):
    env_name = result['env_name']
    if env_name == 'oderl-cancer':
        return plot_telem_cancer(result)
    else:
        return plot_telem_standard(result)

def plot_telem_cancer(result):
    # Cancer
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd
    import seaborn as sn
    pd.set_option('mode.chained_assignment', None)
    SCALE = 13
    HEIGHT_SCALE = 0.5
    LEGEND_Y_CORD = -0.75  # * (HEIGHT_SCALE / 2.0)
    SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
    LEGEND_X_CORD = 0.45
    PLOT_FROM_CACHE = False
    PLOT_SAFTEY_MARGIN = 1.25
    MODEL_NAME_MAP = {}
    time_multiplier = 5.0
    

    plot_actions = True
    plot_reward = True
    plot_var_names = False

    sn.set(rc={'figure.figsize': (SCALE, int(HEIGHT_SCALE * SCALE))})
    sn.set(font_scale=2.0)
    sn.set_style('white', {'font.family':'serif',
                            'font.serif':'Times New Roman',
                            "pdf.fonttype": 42,
                            "ps.fonttype": 42,
                            "font.size": 14})
    sn.color_palette("colorblind")
    plt.gcf().subplots_adjust(bottom=0.20, left=0.2, top=0.9)

    s = result['s'].copy()
    r = result['ri'].copy()
    a = result['a'].copy()
    cost_std_plot = result['cost_std_plot'].copy()
    observed_times = result['observed_times'].copy()
    t = np.arange(s.shape[0]) * result['dt_sim'] * time_multiplier
    observing_var_threshold = result['observing_var_threshold']
    env_name = result['env_name']
    sampling_policy = result['sampling_policy']
    observation_noise = result['observation_noise']
    plot_seed = result['plot_seed']
    if env_name == 'oderl-cancer':
        max_chemo_drug = 5.0
        max_radio = 2.0
        a[:,0] = (a[:,0] / 2.0) * max_chemo_drug
        a[:,1] = (a[:,1] / 2.0) * max_radio
        a[:,0][a[:,0]<=0] = 0
        a[:,1][a[:,1]<=0] = 0
    death_thres = calc_volume(13)
    has_died = (s[:,0] > death_thres).any()

    plots_total = s.shape[1] + 1
    plots_total += a.shape[1] if plot_actions else 0
    plots_total += 1 if plot_reward else 0

    plot_index = 1
    si = 0
    ai = 0

    for plot_index in range(1,plots_total+1):
        if plot_index == 1:
            ax = plt.subplot(plots_total, 1, plot_index)
            plt.title(f'Patient Alive? {not has_died}')
        else:
            ax = plt.subplot(plots_total, 1, plot_index, sharex=ax)
        if si < s.shape[1]:# and si == 0:
            plt.plot(t, s[:,si])
            if si == 0:
                if plot_var_names:
                    plt.ylabel(f'Cancer\nVolume $V$')
                else:
                    plt.ylabel(f'$s_{si}$')
                plt.axhline(y=death_thres, color = 'r') #, linestyle = '-')
                plt.ylim([0,death_thres * 1.2])
            elif si == 1:
                if plot_var_names:
                    plt.ylabel(f'Chemo\nConcentration $C$')
                else:
                    plt.ylabel(f'$s_{si}$')
                plt.ylim([0,10])
            for obs_t in t[observed_times]:
                plt.axvline(x=obs_t, color='g')
            si += 1
        elif plot_actions and ai < a.shape[1]:
            plt.plot(t, a[:,ai])
            if ai == 0:
                if plot_var_names:
                    plt.ylabel(f'Chemo\nDose $C$/day')
                else:
                    plt.ylabel(f'$a_{ai}$')
            elif ai == 1:
                if plot_var_names:
                    plt.ylabel(f'Radio\nDose $S$/day')
                else:
                    plt.ylabel(f'$a_{ai}$')      
            for obs_t in t[observed_times]:
                plt.axvline(x=obs_t, color='g')
            ai += 1
        elif plot_reward and plot_index == (plots_total - 1):
            plt.plot(t, r)
            plt.ylabel('$r$')
            for obs_t in t[observed_times]:
                plt.axvline(x=obs_t, color='g')
        elif plot_index == plots_total:
            if cost_std_plot.size != 0:
                plt.plot(t, cost_std_plot)
                plt.ylabel('$\\sigma(r)$')
                plt.xlabel('Time $t$ (days)')
                for obs_t in t[observed_times]:
                    plt.axvline(x=obs_t, color='g')

    if not os.path.exists("./plots/telem/"):
        os.makedirs("./plots/telem/")

    plt.savefig(f'./plots/telem/telem_{env_name}_{sampling_policy}_{observation_noise}_{plot_seed}.png')
    plt.savefig(f'./plots/telem/telem_{env_name}_{sampling_policy}_{observation_noise}_{plot_seed}.pdf')
    plt.clf()
    file_path = f'./plots/telem/telem_{env_name}_{sampling_policy}_{observation_noise}_{plot_seed}.png'
    print(file_path)
    print(np.median(cost_std_plot))
    return file_path



def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0


def plot_telem_standard(result):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd
    import seaborn as sn
    pd.set_option('mode.chained_assignment', None)
    SCALE = 13
    HEIGHT_SCALE = 0.5
    LEGEND_Y_CORD = -0.75  # * (HEIGHT_SCALE / 2.0)
    SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
    LEGEND_X_CORD = 0.45
    PLOT_FROM_CACHE = False
    PLOT_SAFTEY_MARGIN = 1.25
    MODEL_NAME_MAP = {}

    plot_actions = True
    plot_reward = True

    sn.set(rc={'figure.figsize': (SCALE, int(HEIGHT_SCALE * SCALE))})
    sn.set(font_scale=2.0)
    sn.set_style('white', {'font.family':'serif',
                            'font.serif':'Times New Roman',
                            "pdf.fonttype": 42,
                            "ps.fonttype": 42,
                            "font.size": 14})
    sn.color_palette("colorblind")

    s = result['s'].copy()
    r = result['ri'].copy()
    a = result['a'].copy()
    cost_std_plot = result['cost_std_plot'].copy()
    observed_times = result['observed_times'].copy()
    t = np.arange(s.shape[0]) * result['dt_sim']
    observing_var_threshold = result['observing_var_threshold']
    env_name = result['env_name']
    sampling_policy = result['sampling_policy']
    observation_noise = result['observation_noise']
    plot_seed = result['plot_seed']

    plots_total = s.shape[1] + 1
    plots_total += a.shape[1] if plot_actions else 0
    plots_total += 1 if plot_reward else 0

    plot_index = 1
    si = 0
    ai = 0

    for plot_index in range(1,plots_total+1):
        if plot_index == 1:
            ax = plt.subplot(plots_total, 1, plot_index)
            plt.title(f'{env_name}')
        else:
            ax = plt.subplot(plots_total, 1, plot_index, sharex=ax)
        if si < s.shape[1]:
            plt.plot(t, s[:,si])
            plt.ylabel(f'$s_{si}$')
            for obs_t in t[observed_times]:
                plt.axvline(x=obs_t, color='g')
            si += 1
        elif plot_actions and ai < a.shape[1]:
            plt.plot(t, a[:,ai])
            plt.ylabel(f'$a_{ai}$')
            for obs_t in t[observed_times]:
                plt.axvline(x=obs_t, color='g')
            ai += 1
        elif plot_reward and plot_index == (plots_total - 1):
            plt.plot(t, r)
            plt.ylabel('$r$')
            for obs_t in t[observed_times]:
                plt.axvline(x=obs_t, color='g')
        elif plot_index == plots_total:
            if cost_std_plot.size != 0:
                plt.plot(t, cost_std_plot)
                plt.ylabel('$\\sigma(r)$')
                plt.xlabel('$t$ (days)')
                for obs_t in t[observed_times]:
                    plt.axvline(x=obs_t, color='g')

    if not os.path.exists("./plots/telem/"):
        os.makedirs("./plots/telem/")

    plt.savefig(f'./plots/telem/telem_{env_name}_{sampling_policy}_{observation_noise}_{plot_seed}.png')
    plt.savefig(f'./plots/telem/telem_{env_name}_{sampling_policy}_{observation_noise}_{plot_seed}.pdf')
    plt.clf()
    file_path = f'./plots/telem/telem_{env_name}_{sampling_policy}_{observation_noise}_{plot_seed}.png'
    print(file_path)
    print(np.median(cost_std_plot))
    return file_path

