import torch
import torch.nn.functional as F
from torch.optim import Adam
from rl_modules.models.networks import ActorCritic
from rl_modules.core.utils import compute_gae
from rl_modules.core.potential_functions import POTENTIALS

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, clip_eps=0.2): # TODO: change LR to 3e-4
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps

    def update(self, obs_buf, act_buf, logp_old_buf, adv_buf, ret_buf, epochs=10, batch_size=64):
        """
        Perform several epochs of PPO updates on a rollout batch.

        There is value-function clipping to stabilise critic and gradient clipping to 0.5 L2-norm.
        """
        N = obs_buf.size(0)
        for _ in range(epochs):
            idx_perm = torch.randperm(N)

            for start in range(0, N, batch_size):
                end = start + batch_size
                idx = idx_perm[start:end]

                obs = obs_buf[idx].float()
                act = act_buf[idx].float()
                logp_old = logp_old_buf[idx].float()
                adv = adv_buf[idx].float()
                ret = ret_buf[idx].float()

                # forward pass
                mean, val = self.model(obs)
                std = self.model.log_std.exp()
                dist = torch.distributions.Normal(mean, std)

                # log-prob and entropy
                logp = dist.log_prob(act).sum(axis = -1)
                entropy = dist.entropy().sum(axis = -1)

                # Policy loss clip
                ratio = torch.exp(logp - logp_old)
                clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(ratio * adv, clip_adv).mean()

                # value loss
                with torch.no_grad():
                    val_old = val.detach()
                val_clipped = val_old + torch.clamp(val - val_old, -0.2, 0.2)
                value_loss = F.mse_loss(val_clipped.squeeze(), ret)

                # total loss with c1=0.5 c2=0.01 for entropy
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                # back‑prop & optimisation
                self.optimizer.zero_grad()
                loss.backward()

                # GRADIENT CLIP that I forgor
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()

    def rollout(self, env, max_steps: int =2048, gamma: float=0.99, lam: float=0.95,
                variant: str = 'none',
                decay_steps: int = 300_000,
                global_step: int = 0,
                scale_k: float = 1.0):
        """
        Collect `max_steps` transitions and return (buffers, new_global_step)

        Special arguments:
        ----------
        variant : str
            'none' | 'l2' | 'l2sq' | 'decay'
        decay_steps : int
            Total steps over which alpha decays to 0 (only for 'decay').
        global_step : int
            Current global environment step (helps compute alpha(t)).
        
        Returns:
        ---------
        8-tuple: obs_buf, act_buf, logp_buf, adv_buf, ret_buf, rew_buf, done_buf, global_step
        """
        # -- choose phi(s) --
        is_using_phi = variant in ("l2", "l2sq", "decay")
        phi_key = "l2sq" if variant == "decay" else variant  # decay uses L2 shape
        # ----------

        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

        obs, _ = env.reset()
        phi_prev = POTENTIALS[phi_key](obs) if is_using_phi else 0.0

        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, logp, _ = self.model.get_action(obs_tensor)
                _, value = self.model(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated

            # ---------- reward shaping ----------
            shaped_reward = reward
            if is_using_phi:
                phi_next = POTENTIALS[phi_key](next_obs)

                # decay coefficient alpha(t)
                alpha = 1.0
                if variant == "decay":
                    alpha = max(1.0 - global_step / decay_steps, 0.0)
                elif variant == "scale":
                    shaped_reward = scale_k * reward # scale_k must be passed down

                shaped_reward = reward + alpha * (gamma * phi_next - phi_prev)
                phi_prev = phi_next
            # --------------------------------

            # # ----- DEBUG (remove after testing) ------------------------
            # if global_step < 50: # only first 50 steps overall
            #     print(f"step {global_step:3d}   raw {reward: .3f}   shaped {shaped_reward: .3f}")
            # # -----------------------------------------------------------


            # Add to transition buffers
            obs_buf.append(obs_tensor.squeeze())
            act_buf.append(action.squeeze())
            logp_buf.append(logp)
            rew_buf.append(torch.tensor(shaped_reward).float())
            done_buf.append(torch.tensor(done, dtype=torch.float32))
            val_buf.append(value.squeeze())

            obs = next_obs
            global_step += 1
            if done:
                obs, _ = env.reset()
                phi_prev = POTENTIALS[phi_key](obs) if is_using_phi else 0.0

        # Convert to tensors
        obs_buf = torch.stack(obs_buf)
        act_buf = torch.stack(act_buf)
        logp_buf = torch.stack(logp_buf)
        rew_buf = torch.stack(rew_buf)
        done_buf = torch.stack(done_buf)
        val_buf = torch.stack(val_buf)

        adv_buf, ret_buf = compute_gae(rew_buf, done_buf, val_buf, gamma, lam)

        # Fix for Advantage normalization
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        return obs_buf, act_buf, logp_buf, adv_buf, ret_buf, rew_buf, done_buf, global_step
