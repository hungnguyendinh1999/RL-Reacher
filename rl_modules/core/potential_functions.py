"""
All Phi(s) potential functions live here.
Each must accept `obs` (raw 10-D numpy array) and return a scalar float.
Reacher-v5 observation indices:
0-1: cos(q1, q2)
2-3: sin(q1, q2)
4-5: target (x, y)
6-7: qvel
8-9: target-fingertip vector (x, y)    this is already (target - tip)
"""

import numpy as np

def l2_distance(obs: np.ndarray) -> float:
    vec = obs[8:10]
    return -float(np.linalg.norm(vec, ord=2))

def l2_squared(obs: np.ndarray) -> float:
    vec = obs[8:10]
    return -float(np.dot(vec, vec)) # |vec|^2

# expose a registry for easy lookup
POTENTIALS = {
    "l2": l2_distance, # 'decay' also uses l2
    "l2sq": l2_squared,
}

if __name__ == "__main__":
    dummy = np.zeros(10)
    dummy[8:10] = np.array([0.3, -0.4])
    assert l2_distance(dummy) == -0.5
    assert l2_squared(dummy) == -0.25