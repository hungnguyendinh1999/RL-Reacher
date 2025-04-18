import torch
import json
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

def make_reacher_env(seed=69, render_mode=None):
    env = gym.make("Reacher-v5", render_mode=render_mode)
    env = Monitor(env)  # Tracks episode rewards and lengths
    env.reset()
    return env

def compute_gae(rewards, dones, values, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
    returns = advantages + values
    return advantages, returns

def train(agent, total_iters=20, rollout_len=2048, save_path="ppo_model.pt", stats_path="data/train_log.json"):
    env = make_reacher_env()
    episode_rewards = []

    for it in tqdm(range(total_iters)):
        obs_buf, act_buf, logp_buf, adv_buf, ret_buf = agent.rollout(env, max_steps=rollout_len)
        agent.update(obs_buf, act_buf, logp_buf, adv_buf, ret_buf)
        ep_reward = ret_buf.mean().item()
        # print(f"[Iter {it+1}] Avg Return: {ep_reward:.2f}")
        episode_rewards.append(ep_reward)

    torch.save(agent.model.state_dict(), save_path)
    with open(stats_path, 'w') as f:
        json.dump({"episode_rewards": episode_rewards}, f)

def evaluate(agent, n_episodes=5):
    env = make_reacher_env()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_rew = 0
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.model.get_action(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action.numpy())
            ep_rew += reward
            done = terminated or truncated
        rewards.append(ep_rew)
    print(f"[Eval] Mean Reward: {np.mean(rewards):.2f}")
    return rewards
