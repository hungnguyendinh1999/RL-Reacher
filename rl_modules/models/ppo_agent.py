import torch
import torch.nn.functional as F
from torch.optim import Adam
from rl_modules.models.networks import ActorCritic
from rl_modules.core.utils import compute_gae

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, clip_eps=0.2):
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps

    def update(self, obs_buf, act_buf, logp_old_buf, adv_buf, ret_buf, epochs=10, batch_size=64):
        for _ in range(epochs):
            idxs = torch.randperm(len(obs_buf))
            for start in range(0, len(obs_buf), batch_size):
                end = start + batch_size
                idx = idxs[start:end]

                obs = obs_buf[idx].float()
                act = act_buf[idx].float()
                logp_old = logp_old_buf[idx].float()
                adv = adv_buf[idx].float()
                ret = ret_buf[idx].float()

                mean, val = self.model(obs)
                std = self.model.log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(act).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1)

                ratio = (logp - logp_old).exp()
                clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(ratio * adv, clip_adv).mean()
                value_loss = F.mse_loss(val.squeeze(), ret)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def rollout(self, env, max_steps=2048, gamma=0.99, lam=0.95):
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

        obs, _ = env.reset()
        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, logp, _ = self.model.get_action(obs_tensor)
                _, value = self.model(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated

            obs_buf.append(obs_tensor.squeeze())
            act_buf.append(action.squeeze())
            logp_buf.append(logp)
            rew_buf.append(torch.tensor(reward))
            done_buf.append(torch.tensor(done, dtype=torch.float32))
            val_buf.append(value.squeeze())

            obs = next_obs
            if done:
                obs, _ = env.reset()

        # Convert to tensors
        obs_buf = torch.stack(obs_buf)
        act_buf = torch.stack(act_buf)
        logp_buf = torch.stack(logp_buf)
        rew_buf = torch.stack(rew_buf)
        done_buf = torch.stack(done_buf)
        val_buf = torch.stack(val_buf)

        adv_buf, ret_buf = compute_gae(rew_buf, done_buf, val_buf, gamma, lam)
        return obs_buf, act_buf, logp_buf, adv_buf, ret_buf
