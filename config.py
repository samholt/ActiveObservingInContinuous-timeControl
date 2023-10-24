import numpy as np
import argparse
import torch
import random
import numpy as np
from time import time

def default_config():
    return dict(seed_runs=1000,
        seed_start=0,
        dt=0.1,
        dt_simulation=0.01,
        mppi_roll_outs=1000,
        mppi_time_steps=20,
        mppi_lambda=0.01,
        mppi_sigma=1.0,
        observation_noise=0.01,
        observing_cost=50,
        observing_fixed_frequency=1,
        discrete_planning=True,
        discrete_interval=10,
        continuous_time_threshold=1.0, # From [0,1]
        #==============Expert Dataset collection
        collect_expert_samples=1e6,
        collect_expert_force_generate_new_data=False,
        collect_expert_random_action_noise=1.0,
        collect_expert_cores_per_env_sampler=16,
        collect_expert_episodes_per_sampler_task=1,
        train_with_expert_trajectories=False,
        offline_datasets_path='./offlinedata/',
        #==============Model parameters
        saved_models_path='./saved_models/', 
        normalize=True,
        normalize_time=True,
        model_pe_hidden_units=256,
        encode_obs_time=False,
        model_seed=0,
        #====New parameters
        model_ensemble_size=5,
        model_pe_activation='tanh',
        model_pe_initialization='xavier',
        model_pe_use_pets_log_var=True,
        #==============Training parameters
        weight_decay=0,
        learning_rate=1e-4,
        training_epochs=10000000,
        training_batch_size=16,
        iters_per_log=500,
        clip_grad_norm=0.1,
        clip_grad_norm_on=False,
        train_dt_multiple=1,
        ts_grid='exp', # ['fixed', 'uniform', 'exp']
        train_samples_per_dim=10,
        iters_per_evaluation=1e15,
        lr_scheduler_step_size=20,
        lr_scheduler_gamma=0.1,
        use_lr_scheduler=False,
        reuse_state_actions_when_sampling_times=False,
        end_training_after_seconds=int(1350 * 6.0),
        rand_sample=True,
        #==============Misc
        log_folder='logs',
        save_video=False,
        plot_telem=False,
        sweep_mode=False,
        torch_deterministic=True,
        multi_process_results=True,
        retrain=False,
        force_retrain=False,
        start_from_checkpoint=True,
        print_settings=False,
        training_use_only_samples=None,
        friction=False,
        wandb_project='ActiveObservingControl',
        oracle_var_type='state_oracle_var',
        special_mode_continuous_planning_execute_only_n_actions=None,
        use_95_ci=True,
        )


def parse_args(config):
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_runs", type=int, default=config['seed_runs'], help="seed_runs")
    parser.add_argument("--retrain", choices=('True','False'), default=str(config['retrain']), help="retrain")
    parser.add_argument("--force_retrain", choices=('True','False'), default=str(config['force_retrain']), help="force_retrain")
    parser.add_argument("--start_from_checkpoint", choices=('True','False'), default=str(config['start_from_checkpoint']), help="start_from_checkpoint")
    parser.add_argument("--print_settings", choices=('True','False'), default=str(config['print_settings']), help="print_settings")
    parser.add_argument("--seed_start", type=int, default=config['seed_start'], help="seed_start")
    parser.add_argument("--dt", type=float, default=config['dt'], help="dt")
    parser.add_argument("--learning_rate", type=float, default=config['learning_rate'], help="learning_rate")
    parser.add_argument("--collect_expert_samples", type=float, default=config['collect_expert_samples'], help="collect_expert_samples")
    parser.add_argument("--training_epochs", type=int, default=config['training_epochs'], help="training_epochs")
    parser.add_argument("--training_batch_size", type=int, default=config['training_batch_size'], help="training_batch_size")
    parser.add_argument("--saved_models_path", type=str, default=config['saved_models_path'], help="saved_models_path")
    parser.add_argument("--offline_datasets_path", type=str, default=config['offline_datasets_path'], help="offline_datasets_path")
    parser.add_argument("--iters_per_log", type=int, default=config['iters_per_log'], help="iters_per_log")
    parser.add_argument("--clip_grad_norm", type=float, default=config['clip_grad_norm'], help="clip_grad_norm")
    parser.add_argument("--clip_grad_norm_on", choices=('True','False'), default=str(config['clip_grad_norm_on']), help="clip_grad_norm_on")
    parser.add_argument("--collect_expert_cores_per_env_sampler", type=float, default=config['collect_expert_cores_per_env_sampler'], help="collect_expert_cores_per_env_sampler")
    parser.add_argument("--collect_expert_episodes_per_sampler_task", type=float, default=config['collect_expert_episodes_per_sampler_task'], help="collect_expert_episodes_per_sampler_task")
    parser.add_argument("--normalize", choices=('True','False'), default=str(config['normalize']), help="normalize")
    parser.add_argument("--normalize_time", choices=('True','False'), default=str(config['normalize_time']), help="normalize_time")
    parser.add_argument("--train_dt_multiple", type=float, default=config['train_dt_multiple'], help="train_dt_multiple")
    parser.add_argument("--collect_expert_random_action_noise", type=float, default=config['collect_expert_random_action_noise'], help="collect_expert_random_action_noise")
    parser.add_argument("--ts_grid", type=str, default=config['ts_grid'], help="ts_grid")
    parser.add_argument("--train_samples_per_dim", type=int, default=config['train_samples_per_dim'], help="train_samples_per_dim")
    parser.add_argument("--model_pe_hidden_units", type=int, default=config['model_pe_hidden_units'], help="model_pe_hidden_units")
    parser.add_argument("--lr_scheduler_step_size", type=int, default=config['lr_scheduler_step_size'], help="lr_scheduler_step_size")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=config['lr_scheduler_gamma'], help="lr_scheduler_gamma")
    parser.add_argument("--weight_decay", type=float, default=config['weight_decay'], help="weight_decay")
    parser.add_argument("--log_folder", type=str, default=config['log_folder'], help="log_folder")
    parser.add_argument("--iters_per_evaluation", type=float, default=config['iters_per_evaluation'], help="iters_per_evaluation")
    parser.add_argument("--mppi_roll_outs", type=int, default=config['mppi_roll_outs'], help="mppi_roll_outs")
    parser.add_argument("--mppi_time_steps", type=int, default=config['mppi_time_steps'], help="mppi_time_steps")
    parser.add_argument("--mppi_lambda", type=float, default=config['mppi_lambda'], help="mppi_lambda")
    parser.add_argument("--mppi_sigma", type=float, default=config['mppi_sigma'], help="mppi_sigma")
    parser.add_argument("--encode_obs_time", choices=('True','False'), default=str(config['encode_obs_time']), help="encode_obs_time")
    parser.add_argument("--reuse_state_actions_when_sampling_times", choices=('True','False'), default=str(config['reuse_state_actions_when_sampling_times']), help="reuse_state_actions_when_sampling_times")
    parser.add_argument("--model_seed", type=int, default=config['model_seed'], help="model_seed")
    parser.add_argument("--save_video", choices=('True','False'), default=str(config['save_video']), help="save_video")
    parser.add_argument("--sweep_mode", choices=('True','False'), default=str(config['sweep_mode']), help="sweep_mode")
    parser.add_argument("--rand_sample", choices=('True','False'), default=str(config['rand_sample']), help="rand_sample")
    parser.add_argument("--collect_expert_force_generate_new_data", choices=('True','False'), default=str(config['collect_expert_force_generate_new_data']), help="collect_expert_force_generate_new_data")
    parser.add_argument("--train_with_expert_trajectories", choices=('True','False'), default=str(config['train_with_expert_trajectories']), help="train_with_expert_trajectories")
    parser.add_argument("--end_training_after_seconds", type=float, default=config['end_training_after_seconds'], help="end_training_after_seconds")
    parser.add_argument("--torch_deterministic", choices=('True','False'), default=str(config['torch_deterministic']), help="torch_deterministic")
    parser.add_argument("--use_lr_scheduler", choices=('True','False'), default=str(config['use_lr_scheduler']), help="use_lr_scheduler")
    parser.add_argument("--multi_process_results", choices=('True','False'), default=str(config['multi_process_results']), help="multi_process_results")
    parser.add_argument("--observation_noise", type=float, default=config['observation_noise'], help="observation_noise")
    parser.add_argument("--friction", choices=('True','False'), default=str(config['friction']), help="friction")
    args = parser.parse_args()
    ddict = vars(args)
    ddict['normalize'] = ddict['normalize'] == 'True'
    ddict['normalize_time'] = ddict['normalize_time'] == 'True'
    ddict['encode_obs_time'] = ddict['encode_obs_time'] == 'True'
    ddict['reuse_state_actions_when_sampling_times'] = ddict['reuse_state_actions_when_sampling_times'] == 'True'
    ddict['save_video'] = ddict['save_video'] == 'True'
    ddict['sweep_mode'] = ddict['sweep_mode'] == 'True'
    ddict['rand_sample'] = ddict['rand_sample'] == 'True'
    ddict['collect_expert_force_generate_new_data'] = ddict['collect_expert_force_generate_new_data'] == 'True'
    ddict['train_with_expert_trajectories'] = ddict['train_with_expert_trajectories'] == 'True'
    ddict['torch_deterministic'] = ddict['torch_deterministic'] == 'True'
    ddict['use_lr_scheduler'] = ddict['use_lr_scheduler'] == 'True'
    ddict['multi_process_results'] = ddict['multi_process_results'] == 'True'
    ddict['retrain'] = ddict['retrain'] == 'True'
    ddict['force_retrain'] = ddict['force_retrain'] == 'True'
    ddict['start_from_checkpoint'] = ddict['start_from_checkpoint'] == 'True'
    ddict['print_settings'] = ddict['print_settings'] == 'True'
    ddict['friction'] = ddict['friction'] == 'True'
    ddict['clip_grad_norm_on'] = ddict['clip_grad_norm_on'] == 'True'
    return ddict


def get_config():
    defaults = default_config()
    args = parse_args(defaults)
    defaults.update(args)
    return defaults

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def default_config_dd():
    d_c = default_config()
    return dotdict(d_c)

def seed_all(seed=None):
    """
    Set the torch, numpy, and random module seeds based on the seed
    specified in config. If there is no seed or it is None, a time-based
    seed is used instead and is written to config.
    """
    # Default uses current time in milliseconds, modulo 1e9
    if seed is None:
        seed = round(time() * 1000) % int(1e9)

    # Set the seeds using the shifted seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_model_cuda_memory_details():
    return {'largest_model_pe': 1272, # MiB
            'largest_model_oracle': 2794, #
    }

def load_observing_var_thresholds():
    return {0.05: {'oderl-cartpole': {'continuous': 0.025, 'discrete': 0.25},
            'oderl-pendulum': {'continuous': 0.025, 'discrete': 0.25},
            'oderl-acrobot': {'continuous': 0.025, 'discrete': 0.25},
            'oderl-cancer': {'continuous': 1.0, 'discrete': 1.5},
    },
    0.1: {'oderl-cartpole': {'continuous': 0.029934801, 'discrete': 0.5},
            'oderl-pendulum': {'continuous': 0.012269268, 'discrete': 0.061973028},
            'oderl-acrobot': {'continuous': 0.08927406, 'discrete': 0.28180087},
            'oderl-cancer': {'continuous': 2.9376497, 'discrete': 2.5688453},
    },
    0.2: {'oderl-cartpole': {'continuous': 0.06576242, 'discrete': 0.8842558},
            'oderl-pendulum': {'continuous': 0.04570341, 'discrete': 0.4898505},
            'oderl-acrobot': {'continuous': 0.27656594, 'discrete': 1.6966783},
            'oderl-cancer': {'continuous': 3.657069, 'discrete': 4.8863864},
    },
    0.4: {'oderl-cartpole': {'continuous': 0.19495021, 'discrete': 72.789734},
            'oderl-pendulum': {'continuous': 0.046161246, 'discrete': 1.2940274},
            'oderl-acrobot': {'continuous': 0.82613117, 'discrete': 3.5674138},
            'oderl-cancer': {'continuous': 6.760299902695876, 'discrete': 14.775529274573692},
    }
    }

def load_observing_var_threshold_ranges():
    return {0.1: {'oderl-cartpole': {'continuous': {'lower': 0.0005, 'upper': 0.45}, 'discrete': 0.5}, # Use these ones!
            'oderl-pendulum': {'continuous': {'lower': 0.0005, 'upper': 0.45}, 'discrete': 0.061973028},
            'oderl-acrobot': {'continuous': {'lower': 0.02, 'upper': 5.0}, 'discrete': 0.413525},
            'oderl-cancer': {'continuous': {'lower': 2.0, 'upper': 19.0}, 'discrete': 2.5688453},
    },
    0.4: {'oderl-cancer': {'continuous': {'lower': 2.0, 'upper': 19.0}, 'discrete': 14.775529274573692}}
    }

    