import torch
import json
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

def make_reacher_env(render_mode=None, max_episode_steps=50):
    env = gym.make("Reacher-v5", render_mode=render_mode, max_episode_steps=max_episode_steps)
    env = Monitor(env)  # Tracks episode rewards and lengths
    env.reset()
    return env

def compute_gae(
    rewards: torch.Tensor,
    dones:   torch.Tensor,
    values:  torch.Tensor,
    gamma: float = 0.99,
    lam:   float = 0.95,
):
    """
    Generalized Advantage Estimation (GAE-lambda).

    Parameters
    ----------
    rewards : 1-D Tensor, shape (T,)
        r_t  — raw *per-step* rewards after any shaping.
    dones   : 1-D Tensor, shape (T,)
        d_t  — 1.0 if episode terminated/truncated at step t, else 0.0.
        Used to mask bootstrapping across episode boundaries.
    values  : 1-D Tensor, shape (T,)
        V_t  — critic's value estimates for states s_t (detached from graph).
    gamma   : float
        Discount factor (0 <= gamma <= 1).
    lam     : float
        GAE smoothing parameter lambda (0 <= lambda <= 1).
        lambda = 1 means high variance, low bias (monte-carlo).
        lambda = 0 means low variance, high bias (TD-1 step).

    Returns
    -------
    advantages : Tensor, shape (T,)
        A~_t  — advantage estimates used in PPO's policy loss.
        Computed backward through time:
            sigma_t   =  r_t + gamma*(1-d_t)*V_{t+1}  -  V_t
            A~_t   =  sigma_t + gamma*lamba*(1-d_t)*A~_{t+1}
    returns    : Tensor, shape (T,)
        R_t = A~_t + V_t  — monte-carlo return targets for critic loss.
        (Often called “TD(lamba) returns”.)

    Notes
    -----
    * The `(1 - d_t)` term stops bootstrapping at episode ends so
      future values do not leak across resets.
    * After calling this function, a common practice is to normalize `advantages` across the batch:
            advantages = (advantages - advantages.mean()) / (advantages.std()+1e-8)
      which stabilizes PPO updates.
    """
    advantages = torch.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_adv
    returns = advantages + values
    return advantages, returns

def train(
        agent, 
        total_iters: int=20,
        rollout_len: int=2048,
        variant: str = "none",
        decay_steps: int = 300_000,
        scale_k: float = 1.0, # (Note to self: only used if variant=="scale")
        save_path: str="ppo_model.pt",
        stats_path: str="data/train_log.json"):

    """
    Outer training loop for PPOAgent

    Parameters
    ----------
    agent : PPOAgent
    total_iters : int
        Number of PPO update iterations.
    rollout_len : int
        Steps collected per iteration.
    variant : {'none','l2','l2sq','decay','scale'}
        Reward-shaping mode.
    decay_steps : int
        Horizon where alpha(t) decays to 0 (variant='decay').
    scale_k : float
        Multiplier if variant='scale'. Otherwise just forget it exists
    """
    env = make_reacher_env()
    episode_returns, global_step = [], 0
    running_ep_ret = 0.0
    mean_log = []

    for it in tqdm(range(total_iters), desc=f"{variant}"):
        (obs_buf, act_buf, logp_buf,
         adv_buf, ret_buf,
         rew_buf, done_buf,
         global_step) = agent.rollout(
            env, max_steps = rollout_len, variant = variant,
            decay_steps = decay_steps, global_step = global_step, scale_k = scale_k
        )
        # track per‑episode returns
        for r, d in zip(rew_buf, done_buf):
            running_ep_ret += r.item()
            if d.item() == 1.0:
                episode_returns.append(running_ep_ret)
                running_ep_ret = 0.0

        # PPO update
        agent.update(obs_buf, act_buf, logp_buf, adv_buf, ret_buf)

        # log mean of completed "episodes" (well, this isn't episodic, but it's truncated at 50 steps)
        if episode_returns:
            mean_ep = float(np.mean(episode_returns[-min(10, len(episode_returns)):]))
            mean_log.append(mean_ep)
    
    print(f"[{variant}| {total_iters} iters] last mean ep-return {mean_log[-1]:.2f}")


    torch.save(agent.model.state_dict(), save_path)
    with open(stats_path, 'w') as f:
        json.dump({"episode_returns": episode_returns, "mean_returns": mean_log}, f, indent=2)

    env.close()
