from rl_modules.models.ppo_agent import PPOAgent
from rl_modules.core.utils import train

agent = PPOAgent(obs_dim=10, act_dim=2)
total_iters = 1000
file_suffix = f"{total_iters // 1000}K" if total_iters >= 1000 else f"{total_iters}"
train(agent, total_iters=total_iters, save_path="ppo_model.pt", stats_path=f"data/train_log_{file_suffix}.json")
