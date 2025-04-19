import torch
import numpy as np
from rl_modules.models.networks import ActorCritic
from rl_modules.core.utils import make_reacher_env

def run_rollout(model_path="ppo_model.pt", max_episode_steps=300, render=False, record=False, save_path="rollout.npz"):
    # Set up environment
    render_mode = "rgb_array" if record else "human" if render else None
    env = make_reacher_env(render_mode=render_mode, max_episode_steps=max_episode_steps)
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
        imageio.mimsave("rollout.gif", frames, fps=30)
        print("Saved rollout.gif")

    # Optionally save trajectory
    # np.savez(save_path, trajectory=traj)
    # print(f"Saved rollout trajectory to {save_path}")

    env.close()

if __name__ == "__main__":
    run_rollout(model_path="ppo_model_10K.pt", max_episode_steps=300, render=True, record=False)
