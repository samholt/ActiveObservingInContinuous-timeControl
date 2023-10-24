import numpy as np
import torch
import torch.optim as optim
import os
import time
from config import get_config
from overlay import setup_logger, create_env, generate_irregular_data_time_multi, load_expert_irregular_data_time_multi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.multiprocessing import get_logger
logger = get_logger()

def gaussian_NLL_multi_log_var(y, means, log_variances):
    # https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e confirms
    # Okay up to 6sf
    assert means.shape == log_variances.shape and len(y.shape) == 2,  "error in shapes"
    inv_variances = torch.exp(-log_variances)
    return (torch.sum(log_variances, dim=2) + torch.sum(torch.square(y - means) * inv_variances, dim=2)).mean(dim=1)

def gaussian_NLL_multi(y, means, variances):
    # https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e confirms
    # Okay up to 6sf
    assert means.shape == variances.shape and len(y.shape) == 2,  "error in shapes"
    return (torch.log(torch.prod(variances, dim=2)) + torch.sum(torch.square(y - means) / variances, dim=2)).mean(dim=1)

def get_pe_model(state_dim, action_dim, state_mean, action_mean, state_std, action_std, config, discrete):
    from w_pe import ProbabilisticEnsemble
    return ProbabilisticEnsemble(state_dim,
                                action_dim,
                                hidden_units=config.model_pe_hidden_units,
                                ensemble_size=config.model_ensemble_size,
                                encode_obs_time=config.encode_obs_time,
                                state_mean=state_mean,
                                state_std=state_std,
                                action_mean=action_mean,
                                action_std=action_std,
                                normalize=config.normalize,
                                normalize_time=config.normalize_time,
                                model_activation=config.model_pe_activation,
                                model_initialization=config.model_pe_initialization,
                                model_pe_use_pets_log_var=config.model_pe_use_pets_log_var,
                                discrete=discrete
                                )

def gaussian_NLL_multi_no_matrices(y, mean, variance):
    # https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e confirms
    # Okay up to 6sf
    assert y.shape == mean.shape == variance.shape and len(y.shape) == 2,  "error in shapes"
    return (torch.log(torch.prod(variance, dim=1)) + torch.sum(torch.square(y - mean) / variance, dim=1)).mean()

def train_model(model_name, train_env_task, config, wandb, retrain=False, force_retrain=False, model_seed=0, start_from_checkpoint=False, print_settings=True, evaluate_model_when_trained=False):
    model_saved_name = f'{model_name}_{train_env_task}_ts-grid-{config.ts_grid}-{config.dt}_{model_seed}_train-with-expert-trajectories-{config.train_with_expert_trajectories}_observation-noise-{config.observation_noise}_friction-{config.friction}_model-{config.model_ensemble_size}-{config.model_pe_hidden_units}-log-var-{config.model_pe_use_pets_log_var}'
    if config.end_training_after_seconds is None:
        model_saved_name = f'{model_saved_name}_training_for_epochs-{config.training_epochs}'
    if config.training_use_only_samples is not None:
        model_saved_name = f'{model_saved_name}_samples_used-{config.training_use_only_samples}'
    model_saved_name = f'{model_saved_name}.pt'
    model_path = f'{config.saved_models_path}{model_saved_name}'
    env = create_env(train_env_task, ts_grid=config.ts_grid, dt=config.dt * config.train_dt_multiple, device='cpu')
    obs_state = env.reset()
    state_dim = obs_state.shape[0]
    action_dim = env.action_space.shape[0]

    # logger.info(f'[Test logging when training] {model_name}, {train_env_task}, {config}, {wandb}, {delay}')
    # s0, a0, sn, ts = generate_irregular_data_time_multi(train_env_task, env, samples_per_dim=2, rand=config.rand_sample, delay=delay)
    # if not retrain:
    #     s0, a0, sn, ts = generate_irregular_data_time_multi(train_env_task, env, samples_per_dim=2, rand=config.rand_sample, delay=delay)
    # else:    
    #     s0, a0, sn, ts = generate_irregular_data_time_multi(train_env_task, env, samples_per_dim=15, rand=config.rand_sample, delay=delay)

    # s0, a0, sn, ts = generate_irregular_data_time_multi(train_env_task,
    #                                                 env,
    #                                                 samples_per_dim=config.train_samples_per_dim,
    #                                                 rand=config.rand_sample,
    #                                                 mode=config.ts_grid,
    #                                                 encode_obs_time=config.encode_obs_time,
    #                                                 reuse_state_actions_when_sampling_times=config.reuse_state_actions_when_sampling_times,
    #                                                 observation_noise=config.observation_noise)
    # raise ValueError

    # state_mean = s0.mean(0).detach().cpu().numpy()
    # state_std = s0.std(0).detach().cpu().numpy()
    # action_mean = a0.mean().detach().cpu().numpy()
    # ACTION_HIGH = env.action_space.high[0]
    # action_std = np.array([ACTION_HIGH/2.0])
    
    action_mean = np.array([0]*action_dim)
    ACTION_HIGH = env.action_space.high[0]
    if train_env_task == 'oderl-cartpole':
        state_mean = np.array([0.0,  0.0,  0.0, 0.0, 0.0])
        state_std = np.array([ 2.88646771, 11.54556671,  0.70729307,  0.70692035, 17.3199048 ])
        action_std = np.array([ACTION_HIGH/2.0])
    elif train_env_task == 'oderl-pendulum':
        state_mean = np.array([0.0,  0.0, 0.0])
        state_std = np.array([0.70634571, 0.70784512, 2.89072771])
        action_std = np.array([ACTION_HIGH/2.0])
    elif train_env_task == 'oderl-acrobot':
        state_mean = np.array([0.0,  0.0,  0.0, 0.0, 0.0, 0.0])
        state_std = np.array([0.70711024, 0.70710328, 0.7072186 , 0.7069949 , 2.88642115, 2.88627309])
        action_std = np.array([ACTION_HIGH/2.0])
    elif train_env_task == 'oderl-cancer':
        state_mean = np.array([582.4288,   5.0340])
        state_std = np.array([334.3091,   2.8872])
        action_std = np.array([ACTION_HIGH/2.0])

    if model_name == 'pe':
        model = get_pe_model(state_dim, action_dim, state_mean, action_mean, state_std, action_std, config, discrete=False).to(device)
    elif model_name == 'pe-discrete':
        model = get_pe_model(state_dim, action_dim, state_mean, action_mean, state_std, action_std, config, discrete=True).to(device)
    else:
        raise NotImplementedError
    model_number_of_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f'[{train_env_task}\t{model_name}\tsamples={config.training_use_only_samples}][Model] params={model_number_of_parameters}')

    if not force_retrain:
        logger.info(f'[{train_env_task}\t{model_name}\tsamples={config.training_use_only_samples}]Trying to load : {model_path}')
        if not retrain and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            return model.eval(), {'total_reward': None}
        elif not retrain:
            raise ValueError
        if start_from_checkpoint and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
    if print_settings:
        logger.info(f'[{train_env_task}\t{model_name}\tsamples={config.training_use_only_samples}][RUN SETTINGS]: {config}')
    if wandb is not None:
        wandb.config.update({f"{model_name}__number_of_parameters": model_number_of_parameters}, allow_val_change=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma, verbose=True)
    loss_l = []
    model.train()
    iters = 0

    best_loss = float("inf")
    waiting = 0
    patience = float("inf")

    batch_size = config.training_batch_size
    train_start_time = time.perf_counter()
    elapsed_time = time.perf_counter() - train_start_time
    torch.save(model.state_dict(), model_path)
    if config.train_with_expert_trajectories and config.training_use_only_samples is not None:
        s0, a0, sn, ts = generate_irregular_data_time_multi(train_env_task,
                                                                    encode_obs_time=config.encode_obs_time,
                                                                    config=config)
        permutation = torch.randperm(s0.size()[0])
        permutation = permutation[:config.training_use_only_samples]
    for epoch_i in range(config.training_epochs):
        iters = 0
        nnl_cum_loss = 0
        mse_cum_loss = 0
        t0 = time.perf_counter()
        samples_per_dim = config.train_samples_per_dim
        if config.train_with_expert_trajectories:
            s0, a0, sn, ts = load_expert_irregular_data_time_multi(train_env_task,
                                                                    encode_obs_time=config.encode_obs_time,
                                                                    config=config)
        else:
            s0, a0, sn, ts = generate_irregular_data_time_multi(train_env_task,
                                                                env,
                                                                samples_per_dim=config.train_samples_per_dim,
                                                                rand=config.rand_sample,
                                                                mode=config.ts_grid,
                                                                encode_obs_time=config.encode_obs_time,
                                                                reuse_state_actions_when_sampling_times=config.reuse_state_actions_when_sampling_times,
                                                                observation_noise=config.observation_noise)
        s0, a0, sn, ts = s0.to(device), a0.to(device), sn.to(device), ts.to(device)
        if config.training_use_only_samples is None:
            permutation = torch.randperm(s0.size()[0])
        if int(permutation.size()[0]/batch_size) < config.iters_per_log:
            config.update({'iters_per_log': int(permutation.size()[0]/batch_size)}, allow_val_change=True)
        for iter_i in range(int(permutation.size()[0]/batch_size)):
            optimizer.zero_grad()
            indices = permutation[iter_i*batch_size:iter_i*batch_size+batch_size]
            bs0, ba0, bsn, bts = s0[indices], a0[indices], sn[indices], ts[indices]
            bsd = bsn - bs0
            if config.model_pe_use_pets_log_var:
                means, log_variances = model._forward_ensemble_separate(bs0, ba0, bts)
                losses = gaussian_NLL_multi_log_var(bsd, means, log_variances)
                losses += 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
            else:
                means, variances = model._forward_ensemble_separate(bs0, ba0, bts)
                losses = gaussian_NLL_multi(bsd, means, variances)
            [loss.backward(retain_graph=True) for loss in losses]
            if config.clip_grad_norm_on:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            nnl_cum_loss += losses.mean().item()
            iters += 1
            # Train loss
            mse_losses = torch.square(means - bsd).mean(-1).mean(-1)
            mse_loss = mse_losses.mean(-1)
            mse_cum_loss += mse_loss.item()


            if (permutation.shape[0] == batch_size) or (iter_i % (config.iters_per_log - 1) == 0 and not iter_i == 0):
                nnl_track_loss = nnl_cum_loss / iters
                mse_track_loss = mse_cum_loss / iters
                elapsed_time = time.perf_counter() - train_start_time
                if config.sweep_mode and config.end_training_after_seconds is not None and elapsed_time > config.end_training_after_seconds:
                    logger.info(f'[{train_env_task}\t{model_name}\tsamples={config.training_use_only_samples}]Ending training')
                    break
                logger.info(f'[{config.dt}|{train_env_task}\t{model_name}\tsamples={config.training_use_only_samples}][epoch={epoch_i+1:04d}|iter={iter_i+1:04d}/{int(permutation.size()[0]/batch_size):04d}|t:{int(elapsed_time)}/{config.end_training_after_seconds if config.sweep_mode else 0}] train_nnl={nnl_track_loss}\t| train_mse={mse_track_loss}\t| s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}')
                t0 = time.perf_counter()
                if wandb is not None:
                    wandb.log({"nnl_loss": nnl_track_loss, "mse_loss": mse_track_loss, "epoch": epoch_i, "model_name": model_name, "env_name": train_env_task})
                iters = 0

                # Early stopping procedure
                if nnl_track_loss < best_loss:
                    best_loss = nnl_track_loss
                    torch.save(model.state_dict(), model_path)
                    waiting = 0
                elif waiting > patience:
                    break
                else:
                    waiting += 1
                nnl_cum_loss = 0
                mse_cum_loss = 0
            if iter_i % (config.iters_per_evaluation - 1) == 0 and not iter_i == 0:
                pass
        if config.sweep_mode and config.end_training_after_seconds is not None and elapsed_time > config.end_training_after_seconds:
            break
        if config.use_lr_scheduler:
            scheduler.step()
        loss_l.append(losses.mean().item())

    logger.info(f'[{train_env_task}\t{model_name}\tsamples={config.training_use_only_samples}][Training Finished] model: {model_name} \t|[epoch={epoch_i+1:04d}|iter={iter_i+1:04d}/{int(permutation.size()[0]/batch_size):04d}] train_nnl={nnl_track_loss}\t| train_mse={mse_track_loss}\t| \t| s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}')
    if evaluate_model_when_trained:
        total_reward = evaluate_model(model, model_name, train_env_task, wandb, config, intermediate_run=False)
    else:
        total_reward = None
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    results = {'train_loss': losses.mean().item(), 'best_val_loss': best_loss, 'total_reward': total_reward}
    return model.eval(), results

def evaluate_model(model, model_name, train_env_task, wandb, config, intermediate_run=False):
    if config.sweep_mode and not intermediate_run:
        seed_all(0)
    from mppi_with_model import mppi_with_model_evaluate_single_step
    eval_result = mppi_with_model_evaluate_single_step(model_name=model_name,
                                        env_name=train_env_task,
                                        roll_outs=config.mppi_roll_outs,
                                        time_steps=config.mppi_time_steps,
                                        lambda_=config.mppi_lambda,
                                        sigma=config.mppi_sigma,
                                        dt=config.dt,
                                        encode_obs_time=config.encode_obs_time,
                                        config=config,
                                        model=model,
                                        # save_video=config.save_video,
                                        save_video=False,
                                        intermediate_run=intermediate_run,
                                        )
    total_reward =  eval_result['total_reward']
    logger.info(f'[Evaluation Result] Total reward {total_reward}')
    if wandb is not None:
        wandb.log({"total_reward": total_reward})
    return total_reward

if __name__ == '__main__':
    import wandb, sys
    defaults = get_config()
    defaults['sweep_mode'] = True # Real run settings
    defaults['end_training_after_seconds'] = int(1350 * 6.0 * 100.0)
    defaults['dt'] = 0.1
    wandb.init(config=defaults, project=defaults['wandb_project'] + 'CancerTraining') #, mode="disabled")
    config = wandb.config
    logger = setup_logger(__file__, log_folder=config.log_folder)
    from config import seed_all
    seed_all(0)
    logger.info('Training a model')
    model_name = 'pe' # 'pe-discrete', 'pe'
    train_env_task = 'oderl-acrobot' # 'oderl-cartpole', 'oderl-acrobot', 'oderl-pendulum', 'oderl-cancer'
    train_model(model_name,
                train_env_task,
                config,
                wandb,
                retrain=True,
                force_retrain=True,
                model_seed=0,
                start_from_checkpoint=True,
                print_settings=True)
    logger.info('')
    wandb.finish()