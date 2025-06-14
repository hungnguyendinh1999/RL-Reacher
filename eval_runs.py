"""
Evaluate every run folder under data/ and save results.csv
Columns: run_name, variant, seed, mean_return, success_rate
Success = final distance < 0.02 m
"""
import os
import glob
import re
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from rl_modules.models.networks import ActorCritic
from rl_modules.core.utils import make_reacher_env

# ------------------------------------------------------------
def eval_agent(model_path, n_ep=100):
    env = make_reacher_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    returns, successes = [], 0
    print(model_path)
    for _ in tqdm(range(n_ep)):
        obs, _ = env.reset()
        ep_ret, done, hit_target = 0.0, False, False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = model.get_action(obs_t)
            obs, reward, term, trunc, _ = env.step(action.numpy())
            done = (term or trunc)
            ep_ret = ep_ret + reward

            # Record successes
            
            dist = np.linalg.norm(obs[8:10]) # target‑tip vector
            if (dist < 0.02):
                hit_target = True

        successes += hit_target
        returns.append(ep_ret)
    env.close()
    return np.mean(returns), successes / n_ep
# ------------------------------------------------------------

records = []
for run_dir in glob.glob("data/*/"):
    mdl = os.path.join(run_dir, "ppo_model.pt")
    if not os.path.isfile(mdl):
        continue
    variant = re.split("[_/]", run_dir.rstrip("/"))[1]  # crude parse
    seed_match = re.search(r"seed(\d+)", run_dir)
    seed = int(seed_match.group(1)) if seed_match else 0
    mean_ret, succ = eval_agent(mdl, n_ep=1000)
    records.append(
        dict(run=os.path.basename(run_dir.rstrip("/")),
             variant=variant, seed=seed,
             mean_return=mean_ret, success_rate=succ)
    )
df = pd.DataFrame(records)
filename = "results_bar3.csv"
df.to_csv(filename, index=False)
print(df)
print(f"\nSaved metrics to {filename}")
