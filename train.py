"""
Train PPO (with optional reward-shaping) on Gymnasium's Reacher-v5.

Examples
--------
# 500 k-step baseline run
python scripts/train.py --variant none --timesteps 300000 --run_name baseline

# Potential-based shaping (L2)
python scripts/train.py --variant l2 --timesteps 300000 --run_name pbrs_l2

# L2 shaping with linear decay to step-200k
python scripts/train.py --variant decay --timesteps 300000 \
                        --decay_steps 200000 --run_name pbrs_decay

# Simple reward scaling x5 (ablation)
python scripts/train.py --variant scale --scale_k 5.0 --timesteps 300000 \
                        --run_name reward_scale_5x
"""
import argparse
import os
import json
import math
import time

# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train PPO on Reacher-v5 with optional reward shaping."
    )
    p.add_argument("--variant", choices=["none", "l2", "l2sq", "decay", "scale"],
                   default="none", help="reward-shaping variant")
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="total environment steps across all rollouts")
    p.add_argument("--rollout_len", type=int, default=2048,
                   help="steps collected per PPO batch (on-policy)")
    p.add_argument("--decay_steps", type=int, default=300_000,
                   help="linear-decay horizon for variant='decay'")
    p.add_argument("--scale_k", type=float, default=2.0,
                   help="reward multiplier if variant='scale'")
    p.add_argument("--run_name", type=str, default=None,
                   help="directory name under data/ (auto-timesstamped if omitted)")
    return p.parse_args()

# ----------------------------------------------------------------------
# thin wrapper around core.utils.train
# ----------------------------------------------------------------------
def train_project(args):
    from rl_modules.core.utils import make_reacher_env
    from rl_modules.models.ppo_agent import PPOAgent
    from rl_modules.core.utils import train

    # ------------------------------------------------------------------
    # determine obs / action dims from a temp env
    # ------------------------------------------------------------------
    tmp_env = make_reacher_env()
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    tmp_env.close()

    agent = PPOAgent(obs_dim, act_dim)

    total_iters = math.ceil(args.timesteps / args.rollout_len)
    run_dir = os.path.join("data", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # save config for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # start training
    train(
        agent=agent,
        total_iters=total_iters,
        rollout_len=args.rollout_len,
        variant=args.variant,
        decay_steps=args.decay_steps,
        scale_k=args.scale_k,
        save_path=os.path.join(run_dir, "ppo_model.pt"),
        stats_path=os.path.join(run_dir, "train_log.json"),
    )

# ----------------------------------------------------------------------
# main entry
def main():
    args = parse_args()

    # auto timestamp if run_name omitted
    if args.run_name is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.variant}_{ts}"

    train_project(args)


if __name__ == "__main__":
    main()
