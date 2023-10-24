import traceback
import torch
import wandb
from config import get_config, seed_all
from torch import multiprocessing
import logging
from functools import partial
from tqdm import tqdm
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAINABLE_MODELS = ['pe', 'pe-discrete']
ENVIRONMENTS = ['oderl-pendulum', 'oderl-cartpole', 'oderl-cancer', 'oderl-acrobot']
SAMPLING_POLICIES = ['discrete_monitoring', 'discrete_planning', 'continuous_planning', 'active_observing_control']
RETRAIN = False
FORCE_RETRAIN = False
START_FROM_CHECKPOINT = False
MODEL_TRAIN_SEED = 0
PRINT_SETTINGS = False

from mppi_with_model_active_observing import mppi_with_model_evaluate_single_step_active_observing
from train_utils import train_model

def train_model_wrapper(args, **kwargs):
    try:
        (env_name, model_name) = args
        from config import seed_all, dotdict
        config = kwargs['config']
        config = dotdict(config)
        kwargs['config'] = config
        logger = create_logger_in_process(config.log_path)
        logger.info(f'[Now training model] {model_name} \t {env_name}')
        seed_all(config.seed_start)
        model, results = train_model(
                model_name,
                env_name,
                **kwargs)
        results['errored'] = False
    except Exception as e:
        logger.exception(f'[Error] {e}')
        logger.info(f"[Failed training model] {env_name} {model_name} delay={delay} \t model_seed={MODEL_TRAIN_SEED} \t | error={e}")
        traceback.print_exc()
        results = {'errored': True}
        print('')
    results.update({'model_name': model_name, 'env_name': env_name})
    logger.info(f'[Training Result] {model_name} result={results}')
    return results


def mppi_with_model_evaluate_single_step_wrapper(args, **kwargs):
    try:
        (env_name, model_name, threshold_percent, sampling_policy, seed)  = args
        from config import seed_all, dotdict
        seed_all(seed)
        config = kwargs['config']
        config = dotdict(deepcopy(config))
        config.observing_var_threshold = threshold_percent
        kwargs['config'] = config
        logger = create_logger_in_process(config.log_path)
        logger.info(f'[Now evaluating policy] {(env_name, model_name, threshold_percent, sampling_policy, seed)}')
        results = mppi_with_model_evaluate_single_step_active_observing(model_name=model_name,
                                                    env_name=env_name,
                                                    sampling_policy=sampling_policy,
                                                    seed=seed,
                                                    planner='mppi_active_observing',
                                                    **kwargs)
        results['errored'] = False
    except Exception as e:
        logger.exception(f'[Error] {e}')
        logger.info(f"[Failed evaluating policy] {(env_name, model_name, threshold_percent, sampling_policy, seed)}\t| error={e}")
        traceback.print_exc()
        results = {'errored': True}
        print('')
    results.update({'model_name': model_name, 'env_name': env_name, 'seed': seed, 'observing_var_threshold': threshold_percent, 'sampling_policy': sampling_policy})
    return results

def train_all_models(config, wandb=None):
    # Re-train all the models
    model_training_results_l = []

    pool_outer = multiprocessing.Pool(config.collect_expert_cores_per_env_sampler)
    if config.retrain:
        train_all_model_inputs = [(env_name, model_name) for env_name in ENVIRONMENTS for model_name in TRAINABLE_MODELS]
        logger.info(f'Going to train for {len(train_all_model_inputs)} tasks')
        with multiprocessing.Pool(len(train_all_model_inputs)) as pool_outer: # 12, or len(train_all_model_inputs) if GPU memory supports training all together.
            multi_wrapper_train_model = partial(train_model_wrapper,
                                                config=dict(config),
                                                wandb=None,
                                                model_seed=config.model_seed,
                                                retrain=config.retrain,
                                                start_from_checkpoint=config.start_from_checkpoint,
                                                force_retrain=config.force_retrain,
                                                print_settings=config.print_settings,
                                                evaluate_model_when_trained=False)
            for i, result in tqdm(enumerate(pool_outer.imap_unordered(multi_wrapper_train_model, train_all_model_inputs)), total=len(train_all_model_inputs), smoothing=0):
                logger.info(f'[Model Completed training] {result}')
                model_training_results_l.append(result)

    # Tune the thresholds manually following method as described in the paper.
    logger.info('[ACTION ITEM] Please now tune the thresholds manually following method as described in the paper.')

def generate_log_file_path(file, log_folder='logs'):
    import os, time, logging
    file_name = os.path.basename(os.path.realpath(file)).split('.py')[0]
    from pathlib import Path
    Path(f"./{log_folder}").mkdir(parents=True, exist_ok=True)
    path_run_name = '{}-{}'.format(file_name, time.strftime("%Y%m%d-%H%M%S"))
    return f"{log_folder}/{path_run_name}_log.txt"

def create_logger_in_process(log_file_path):
    logger = multiprocessing.get_logger()
    if not logger.hasHandlers():
        formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file_path)
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":
    log_path = generate_log_file_path(__file__)
    logger = create_logger_in_process(log_path)
    defaults = get_config()
    defaults['log_path'] = log_path
    defaults['multi_process_results'] = True # debug mode
    if defaults['multi_process_results']:
        torch.multiprocessing.set_start_method('spawn')
    defaults['retrain'] = RETRAIN
    defaults['force_retrain'] = FORCE_RETRAIN
    defaults['start_from_checkpoint'] = START_FROM_CHECKPOINT
    defaults['print_settings'] = PRINT_SETTINGS
    defaults['model_train_seed'] = MODEL_TRAIN_SEED
    defaults['sweep_mode'] = True # Real run settings

    # Test run settings
    defaults['end_training_after_seconds'] = int(1350 * 6.0 * 100.0) # Train for a long time, i.e. until models converge. Usually after three days per model.

    wandb.init(config=defaults, project=defaults['wandb_project']) #, mode="disabled")
    config = wandb.config
    seed_all(0)
    logger.info(f'Starting run \t | See log at : {log_path}')
    train_all_models(config, wandb)
    wandb.finish()
    logger.info('Run over. Fin.')
    logger.info(f'[Log found at] {log_path}')
