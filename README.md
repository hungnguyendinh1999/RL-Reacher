# Reward‑Shaping‑for‑Robotic‑Control  
*A comparative study of reward‑shaping techniques for PPO on Reacher‑v5*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Project Overview
This repo investigates how different **reward‑shaping strategies** influence learning speed and final performance of a PPO agent controlling the 2‑DoF **Reacher‑v5** robotic arm ([Gymnasium](https://gymnasium.farama.org/environments/mujoco/reacher/)).  
We compare:

| Label | Strategy | $\phi$‑function (potential) | Notes |
|-------|----------|------------------------|-------|
| **B‑0** | Baseline | — | raw env reward |
| **$\Delta‑1$** | PBRS‑L2 | $-‖\text{tip}−\text{target}‖$ |
| **$\Delta‑2$** | PBRS‑L2² | $-‖\text{tip}−\text{target}‖^{2}$ |
| **$\Delta‑3$** | PBRS‑Decay | $\Delta‑1 × \alpha(t)$ (linear decay $\rightarrow$ 0) |
| *(todo.)* | Reward‑Scale | $k \cdot r$ | ablation |

---

## 🗂️ Directory Structure
```
Reward-shaping-for-robotic-control/
│
├─ rl_modules/                # importable package
│   ├─ __init__.py
│   ├─ core/
│   │   ├─ __init__.py
│   │   ├─ potential_functions.py # registry of POTENTIAL functions L2, L2_squared
│   │   └─ utils.py           # make_reacher_env(), train(), eval_agent(), GAE, logging
│   └─ models/
│       ├─ __init__.py
│       ├─ networks.py        # Actor‑Critic MLP
│       └─ ppo_agent.py       # PPO with optional shaping (none, L2, L2_squared, decay)
│
├─ data/                      # logs (json), rollouts, models (.pt)
├─ pilot_none/
│   ├─ ppo_model.pt
│   └─ train_log.json
├─ pilot_l2/
│   ├─ ppo_model.pt
│   └─ train_log.json
├─ ...                    # some more data stuff
│
├─ train.py               # command‑line training entry
├─ run_rollout.py         # visualize a saved model
├─ plot_results.ipynb     # analysis / figures
├─ plot_training.py       # plot script for training data
│
│
└─ README.md
```

---

## ⚙️ Installation

```bash
# 1. (Recommended) create a fresh env
conda create -n rl_env python=3.11 -y
conda activate rl_env

# 2. Install dependencies
pip install -r requirements.txt
#  gymnasium[mujoco] for Reacher‑v5, torch, imageio, etc.
```

---

## Quick Start

### 1. Baseline PPO
```bash
python train.py --variant none --timesteps 100000 --run_name baseline
```

### 2. Potential‑based L2 shaping
```bash
python train.py --variant l2 --timesteps 100000 --run_name pbrs_l2
```

### 3. Rollout a trained model
```bash
python run_rollout.py
```
Creates `rollout.gif` and `rollout.npz`.

---

## 🛠️ CLI Options (`train.py`)
| flag | default | description |
|------|---------|-------------|
| `--variant` | `none` | `none`, `l2`, `l2sq`, `decay`, `scale` |
| `--timesteps` | `500000` | total env steps |
| `--decay_steps` | `300k` | when $\alpha(t)=0$ (for `decay`) |
| `--scale_k` | `2.0` | reward multiplier (`scale`) |
| `--run_name` | auto | sub‑folder in `data/` |

---


Questions or contributions? Feel free to open an Issue or PR!
