import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.policy_mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable log std

        self.value = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.shared(obs)
        return self.policy_mean(x), self.value(x)

    def get_action(self, obs):
        mean, _ = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)  # log prob of sampled action
        entropy = dist.entropy().sum(axis=-1)      # entropy bonus (for exploration)
        return action[0], logp, entropy
