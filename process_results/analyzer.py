import random
from time import time

import numpy as np
import torch
from plot_util import df_from_log, generate_main_results_table, normalize_means


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


seed_all(0)


def extract_state_rewards(df):
    dd = {}
    for _, row in df.iterrows():
        k, v = row["observations_taken"], row["state_reward"]
        if k in dd:
            dd[k].append(v)
        else:
            dd[k] = [v]
    return dd


# Print main table of results now
df = df_from_log("./results/main_paper_results.txt")
df = normalize_means(df)
df_out, table_str = generate_main_results_table(df)
print("")
print(table_str)
print("fin.")
