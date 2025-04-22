"""
DEMO: Generate a rollout GIF, or visualize on a new window. Please make appropriate modifications before running.
"""

import glob
import os
from run_rollout import run_rollout
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
        description="Run DEMO PPO on Reacher-v5"
    )
    p.add_argument("--mode", choices=["record", "render"],
                   default="render", help="mode of render - by Recording and saving to a GIF file, or render on new visual window")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for run_dir in glob.glob("demo/*/"):
        model_path = os.path.join(run_dir, "ppo_model.pt")
        if not os.path.isfile(model_path):
            continue
        run_id = run_dir.split("/")[1]
        print("[Run ID]", run_id)

        isRecord = (args.mode == "record")
        run_rollout(
            model_path=model_path,
            render=True,
            record=isRecord, save_gif_path=f"demo/{run_id}.gif" # uncomment this to save GIFs
            )
