"""
Plot mean‑episode‑return curves for every run.
Assumes each train_log.json has {"episode_returns": [...]}.
"""
import os
import glob
import json
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
for run_dir in glob.glob("data/*/"):
    log = os.path.join(run_dir, "train_log.json")
    if not os.path.isfile(log):
        continue
    with open(log) as f:
        ep_rets = json.load(f)["episode_returns"]
    if len(ep_rets) < 2:
        continue
    label = os.path.basename(run_dir.rstrip("/"))
    plt.plot(ep_rets, label=label, alpha=0.8)

plt.title("Reacher-v5 | Episode return vs. PPO iteration")
plt.xlabel("PPO iteration")
plt.ylabel("Episode return")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Saved training_curves.png")
