from tqdm import tqdm
import ast
import pandas as pd
import numpy as np
from scipy import stats

def ci(data, confidence=0.95):
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE, use_autolayout=True):
    pd.set_option('mode.chained_assignment', None)
    sn.set(rc={'figure.figsize': (SCALE, int(HEIGHT_SCALE * SCALE)), 'figure.autolayout': use_autolayout, 'text.usetex': True, 'text.latex.preamble': [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
                    ]  
    })
    sn.set(font_scale=2.0)
    sn.set_style('white', {'font.family':'serif',
                            'font.serif':'Times New Roman',
                            "pdf.fonttype": 42,
                            "ps.fonttype": 42,
                            "font.size": 14})
    sn.color_palette("colorblind")
    return sn

def df_from_log(path, remove_extra_columns=True):
    with open(path) as f:
        lines = f.readlines()
    pd_l = []
    for line in tqdm(lines):
        if '[Policy evaluation complete] {' in line:
            result_dict = line.split('[Policy evaluation complete] ')[1].strip()
            result_dict = result_dict.replace('nan', '\'nan\'')
            result_dict = result_dict.replace('array', '')
            result_dict = ast.literal_eval(result_dict)
            pd_l.append(result_dict)

    dfm = pd.DataFrame(pd_l)
    if remove_extra_columns:
        columns_to_remove_if_exist = ['costs_std_stats', 'planner', 'observed_times', 'observed_times_diff', 'costs_std_median', 's', 'a', 'r', 'cost_std_plot', 'ri', 'telem_file_path']
        current_columns = list(dfm.columns)
        columns_to_drop = set(columns_to_remove_if_exist) & set(current_columns)
        columns_to_drop = list(columns_to_drop)
        dfm = dfm.drop(columns=columns_to_drop)
    else:
        columns_to_np_arrays_if_exist = ['observed_times', 'observed_times_diff', 's', 'a', 'r', 'cost_std_plot', 'ri']
        current_columns = list(dfm.columns)
        columns_to_np_arrays = set(columns_to_np_arrays_if_exist) & set(current_columns)
        columns_to_np_arrays = list(columns_to_np_arrays)
        dfm[columns_to_np_arrays] = dfm[columns_to_np_arrays].applymap(np.array)
    numeric_columns = ['roll_outs',
                        'time_steps',
                        'episode_elapsed_time',
                        'episode_elapsed_time_per_it',
                        'dt_sim',
                        'dt_plan',
                        'total_reward',
                        'state_reward',
                        'state_reward_std',
                        'observation_reward',
                        'observations_taken',
                        'observing_var_threshold',
                        'observing_cost',
                        'observation_noise',
                        'seed']
    dfm[numeric_columns] = dfm[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return dfm

def normalize_means(df):
    df_means = df.groupby(['env_name', 'sampling_policy', 'model_name']).agg(np.mean).reset_index()
    for env_name in df_means.env_name.unique():
        pass
        df_means_env = df_means[df_means.env_name == env_name]
        random_row = df_means_env[df_means_env.sampling_policy == 'random'].iloc[0]
        best_row = df_means_env[df_means_env.sampling_policy == 'continuous_planning'].iloc[0]

        df.loc[df.env_name==env_name, 'total_reward'] = ((df[df.env_name == env_name].total_reward - random_row.total_reward) / (best_row.total_reward - random_row.total_reward)) * 100.0
        df.loc[df.env_name==env_name, 'state_reward'] = ((df[df.env_name == env_name].state_reward - random_row.state_reward) / (best_row.state_reward - random_row.state_reward)) * 100.0
    return df

def normalize_means_cp_matching_obs(df_in):
    df = df_in[df_in.fixed_continuous_planning_observations.isnull()]
    df_means = df.groupby(['env_name', 'sampling_policy', 'model_name']).agg(np.mean).reset_index()
    for env_name in df_means.env_name.unique():
        df_means_env = df_means[df_means.env_name == env_name]
        random_row = df_means_env[df_means_env.sampling_policy == 'random'].iloc[0]
        best_row = df_means_env[df_means_env.sampling_policy == 'continuous_planning'].iloc[0]

        df_in.loc[df_in.env_name==env_name, 'total_reward'] = ((df_in[df_in.env_name == env_name].total_reward - random_row.total_reward) / (best_row.total_reward - random_row.total_reward)) * 100.0
        df_in.loc[df_in.env_name==env_name, 'state_reward'] = ((df_in[df_in.env_name == env_name].state_reward - random_row.state_reward) / (best_row.state_reward - random_row.state_reward)) * 100.0
    return df_in

def remove_unneeded_columns(df):
    columns_to_remove_if_exist = ['costs_std_stats', 'planner', 'observed_times', 'observed_times_diff', 'costs_std_median', 's', 'a', 'r', 'cost_std_plot', 'ri', 'telem_file_path']
    current_columns = list(df.columns)
    columns_to_drop = set(columns_to_remove_if_exist) & set(current_columns)
    columns_to_drop = list(columns_to_drop)
    df = df.drop(columns=columns_to_drop)
    return df

def generate_main_results_table_cp_with_unique_obs(df_results, wandb=None):
    # Process seeds here
    df_results = remove_unneeded_columns(df_results)
    df_out = df_results.groupby(['env_name', 'sampling_policy', 'model_name']).agg([np.mean, np.std]).reset_index()
    df_out.loc[df_out.sampling_policy == 'random', 'total_reward'] = 0
    df_out.loc[df_out.sampling_policy == 'random', 'state_reward'] = 0

    sf = 3
    env_name_map = {'oderl-cartpole': 'Cartpole',
                    'oderl-pendulum': 'Pendulum',
                    'oderl-acrobot': 'Acrobot',
                    'oderl-cancer': 'Cancer'}

    env_name_ordering = {'oderl-cartpole': 2,
                    'oderl-pendulum': 3,
                    'oderl-acrobot': 1,
                    'oderl-cancer': 0}

    sampling_policy_map = {'discrete_monitoring': 'Discrete Monitoring',
                             'discrete_planning': 'Discrete Planning',
                             'continuous_planning': 'Continuous Planning',
                             'active_observing_control': r'\bf Active Sampling Control',
                             'random': 'Random'}
                             
    df_out = df_out.sort_values(by=['env_name'], key=lambda x: x.map(env_name_ordering))
    table_lines = []
    line = r'\begin{tabular}{@{}l' + '|ccc' * df_out.env_name.nunique() + '}'
    table_lines.append(line)
    table_lines.append(r'\toprule')
    table_lines.append(''.join([r'&  \multicolumn{3}{c|}{' + env_name_map[env_name] + '}' for env_name in df_out.env_name.unique()]) + r'\\')
    table_lines.append(r'Policy ' + r'& $\mathcal{U}$ & $\mathcal{R}$ & $\mathcal{O}$' * df_out.env_name.nunique() + r'\\' )
    table_lines.append(r'\midrule')
    for sampling_policy in df_out.sampling_policy.unique():
        line = sampling_policy_map[sampling_policy]
        for env_name in df_out.env_name.unique():
            row = df_out[(df_out.sampling_policy == sampling_policy) & (df_out.env_name == env_name)]
            line += r'&' + f"{row.total_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.total_reward['std'].iloc[0]:.{sf}g}" + r'&' + f"{row.state_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.state_reward['std'].iloc[0]:.{sf}g}" + r'&' + f"{row.observations_taken['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.observations_taken['std'].iloc[0]:.{sf}g}"
        line += r'\\'
        table_lines.append(line)
    table_lines.append(r'\bottomrule')
    table_lines.append(r'\end{tabular}')
    table = '\n'.join(table_lines)
    return df_out, table

def generate_main_results_table(df_results, wandb=None, use_95_ci=True):
    # Process seeds here
    df_results = remove_unneeded_columns(df_results)
    if use_95_ci:
        df_out = df_results.groupby(['env_name', 'sampling_policy', 'model_name']).agg([np.mean, ci]).reset_index()
        error_metric = 'ci'
    else:
        df_out = df_results.groupby(['env_name', 'sampling_policy', 'model_name']).agg([np.mean, np.std]).reset_index()
        error_metric = 'std'
    df_out.loc[df_out.sampling_policy == 'random', 'total_reward'] = 0
    df_out.loc[df_out.sampling_policy == 'random', 'state_reward'] = 0

    sf = 3
    env_name_map = {'oderl-cartpole': 'Cartpole',
                    'oderl-pendulum': 'Pendulum',
                    'oderl-acrobot': 'Acrobot',
                    'oderl-cancer': 'Cancer'}

    env_name_ordering = {'oderl-cartpole': 2,
                    'oderl-pendulum': 3,
                    'oderl-acrobot': 1,
                    'oderl-cancer': 0}

    sampling_policy_map = {'discrete_monitoring': 'Discrete Monitoring',
                             'discrete_planning': 'Discrete Planning',
                             'continuous_planning': 'Continuous Planning',
                             'active_observing_control': r'\bf Active Sampling Control',
                             'random': 'Random'}
                             
    df_out = df_out.sort_values(by=['env_name'], key=lambda x: x.map(env_name_ordering))
    table_lines = []
    line = r'\begin{tabular}{@{}l' + '|ccc' * df_out.env_name.nunique() + '}'
    table_lines.append(line)
    table_lines.append(r'\toprule')
    table_lines.append(''.join([r'&  \multicolumn{3}{c|}{' + env_name_map[env_name] + '}' for env_name in df_out.env_name.unique()]) + r'\\')
    table_lines.append(r'Policy ' + r'& $\mathcal{U}$ & $\mathcal{R}$ & $\mathcal{O}$' * df_out.env_name.nunique() + r'\\' )
    table_lines.append(r'\midrule')
    for sampling_policy in df_out.sampling_policy.unique():
        if sampling_policy == 'active_observing_control':
            line = r'\midrule' + '\n' + sampling_policy_map[sampling_policy]
        else:
            line = sampling_policy_map[sampling_policy]
        for env_name in df_out.env_name.unique():
            row = df_out[(df_out.sampling_policy == sampling_policy) & (df_out.env_name == env_name)]
            if sampling_policy == 'active_observing_control':
                line += (r'&\bf' if row.total_reward['mean'].iloc[0] > 100.0 else r'&') + f"{row.total_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.total_reward[error_metric].iloc[0]:.{sf}g}" + (r'&\bf' if row.state_reward['mean'].iloc[0] > 100.0 else r'&') + f"{row.state_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.state_reward[error_metric].iloc[0]:.{sf}g}" + r'&' + f"{row.observations_taken['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.observations_taken[error_metric].iloc[0]:.{sf}g}"
            else:
                line += r'&' + f"{row.total_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.total_reward[error_metric].iloc[0]:.{sf}g}" + r'&' + f"{row.state_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.state_reward[error_metric].iloc[0]:.{sf}g}" + r'&' + f"{row.observations_taken['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.observations_taken[error_metric].iloc[0]:.{sf}g}"
        line += r'\\'
        table_lines.append(line)
    table_lines.append(r'\bottomrule')
    table_lines.append(r'\end{tabular}')
    table = '\n'.join(table_lines)
    return df_out, table



def gen_match_cp_obs_table(df, use_95_ci=True):
    print('')
    cp_with_different_obs = df[~df.fixed_continuous_planning_observations.isnull()]
    aoc_with_normalization_reference_baselines = df[df.fixed_continuous_planning_observations.isnull()]

    if use_95_ci:
        df_out = aoc_with_normalization_reference_baselines.groupby(['env_name', 'sampling_policy', 'model_name', 'observing_var_threshold']).agg([np.mean, ci]).reset_index()
        cp_dif_obs = cp_with_different_obs.groupby(['env_name', 'sampling_policy', 'model_name', 'fixed_continuous_planning_observations']).agg([np.mean, ci]).reset_index()
    else:
        df_out = aoc_with_normalization_reference_baselines.groupby(['env_name', 'sampling_policy', 'model_name', 'observing_var_threshold']).agg([np.mean, np.std]).reset_index()
        cp_dif_obs = cp_with_different_obs.groupby(['env_name', 'sampling_policy', 'model_name', 'fixed_continuous_planning_observations']).agg([np.mean, np.std]).reset_index()

    df_out.loc[df_out.sampling_policy == 'random', 'total_reward'] = 0
    df_out.loc[df_out.sampling_policy == 'random', 'state_reward'] = 0


    sf = 3
    env_name_map = {'oderl-cartpole': 'Cartpole',
                    'oderl-pendulum': 'Pendulum',
                    'oderl-acrobot': 'Acrobot',
                    'oderl-cancer': 'Cancer'}

    env_name_ordering = {'oderl-cartpole': 2,
                    'oderl-pendulum': 3,
                    'oderl-acrobot': 1,
                    'oderl-cancer': 0}

    sampling_policy_map = {'discrete_monitoring': 'Discrete Monitoring',
                             'discrete_planning': 'Discrete Planning',
                             'continuous_planning': 'Continuous Planning',
                             'active_observing_control': r'\bf Active Sampling Control',
                             'random': 'Random'}
                             
    table_lines = []
    line = r'\begin{tabular}{@{}l' + '|ccc' * df_out.env_name.nunique() + '}'
    table_lines.append(line)
    table_lines.append(r'\toprule')
    table_lines.append(''.join([r'&  \multicolumn{3}{c|}{' + env_name_map[env_name] + '}' for env_name in df_out.env_name.unique()]) + r'\\')
    table_lines.append(r'Policy ' + r'& $\mathcal{U}$ & $\mathcal{R}$ & $\mathcal{O}$' * df_out.env_name.nunique() + r'\\' )
    table_lines.append(r'\midrule')
    for sampling_policy in df_out.sampling_policy.unique():
        line = sampling_policy_map[sampling_policy]
        for env_name in df_out.env_name.unique():
            row = df_out[(df_out.sampling_policy == sampling_policy) & (df_out.env_name == env_name)]
            if use_95_ci:
                line += r'&' + f"{row.total_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.total_reward['ci'].iloc[0]:.{sf}g}" + r'&' + f"{row.state_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.state_reward['ci'].iloc[0]:.{sf}g}" + r'&' + f"{row.observations_taken['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.observations_taken['ci'].iloc[0]:.{sf}g}"
            else:
                line += r'&' + f"{row.total_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.total_reward['std'].iloc[0]:.{sf}g}" + r'&' + f"{row.state_reward['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.state_reward['std'].iloc[0]:.{sf}g}" + r'&' + f"{row.observations_taken['mean'].iloc[0]:.{sf}g}" + r'$\pm$' + f"{row.observations_taken['std'].iloc[0]:.{sf}g}"
        line += r'\\'
        table_lines.append(line)
    for i, row in cp_dif_obs.iterrows():
        line = sampling_policy_map[row.sampling_policy.iloc[0]]
        for env_name in cp_dif_obs.env_name.unique():
            if use_95_ci:
                line += r'&' + f"{row.total_reward['mean']:.{sf}g}" + r'$\pm$' + f"{row.total_reward['ci']:.{sf}g}" + r'&' + f"{row.state_reward['mean']:.{sf}g}" + r'$\pm$' + f"{row.state_reward['ci']:.{sf}g}" + r'&' + f"{row.observations_taken['mean']:.{sf}g}" + r'$\pm$' + f"{row.observations_taken['ci']:.{sf}g}"
            else:
                line += r'&' + f"{row.total_reward['mean']:.{sf}g}" + r'$\pm$' + f"{row.total_reward['std']:.{sf}g}" + r'&' + f"{row.state_reward['mean']:.{sf}g}" + r'$\pm$' + f"{row.state_reward['std']:.{sf}g}" + r'&' + f"{row.observations_taken['mean']:.{sf}g}" + r'$\pm$' + f"{row.observations_taken['std']:.{sf}g}"
        line += r'\\'
        table_lines.append(line)
    table_lines.append(r'\bottomrule')
    table_lines.append(r'\end{tabular}')
    table = '\n'.join(table_lines)
    return df_out, table

def plot_telem_standard(result):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd
    import seaborn as sn
    import os
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

    sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)

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
