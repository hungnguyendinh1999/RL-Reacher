"""
Generate a rollout GIF, or visualize on a new window. Please make appropriate modifications before running.
"""

import torch
import numpy as np
import glob
import os
from rl_modules.models.networks import ActorCritic
from rl_modules.core.utils import make_reacher_env


def run_rollout(model_path="ppo_model.pt", max_episode_steps=50, render=False, record=False, 
                save_gif_path = "rollout.gif", save_traj_path=None):
    """
    Run rollout and generate a new window, rendering the rollout, or generate a rollout.npz file
    `record` and `render` have a hierachical relationship, where
        `record = True` is prioritized. Otherwise, we consider `render=True` to renders in "human" mode
    """
    # Set up environment
    render_mode = "rgb_array" if record else "human" if render else None
    env = make_reacher_env(render_mode=render_mode, max_episode_steps=max_episode_steps)
    env.metadata["render_fps"] = 30
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load model
    model = ActorCritic(obs_dim, act_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    obs, _ = env.reset()
    done = False
    total_reward = 0
    frames = []
    traj = []

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.get_action(obs_tensor)
        obs, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        total_reward += reward
        traj.append((obs, reward))

        if record:
            frame = env.render()
            frames.append(frame)

    print(f"[Rollout] Total reward: {total_reward:.2f}")

    if record:
        import imageio
        imageio.mimsave(save_gif_path, frames, fps=30)
        print(f"Saved {save_gif_path}")

    # Optionally save trajectory
    if save_traj_path:
        np.savez(save_traj_path, trajectory=traj)
        print(f"Saved rollout trajectory to {save_traj_path}")

    env.close()

if __name__ == "__main__":
    for run_dir in glob.glob("data/*/"):
        model_path = os.path.join(run_dir, "ppo_model.pt")
        if not os.path.isfile(model_path):
            continue
        run_id = run_dir.split("/")[2]
        run_rollout(
            model_path=model_path,
            render=True,
            # record=True, save_gif_path=f"{run_id}.gif"
            )
