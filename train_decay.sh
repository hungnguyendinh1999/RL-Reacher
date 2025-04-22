# ---------------------------------------------------------
# launch full 500 k‑step or 1M-step experiments
# ---------------------------------------------------------
set -e          # exit on first error
TIMESTEPS=1500
ROLLOUT=50
LABEL=50_1500
# ---- loops ------------------------------------------------

#   baseline
  python train.py \
      --variant none \
      --timesteps $TIMESTEPS \
      --rollout_len $ROLLOUT \
      --run_name none_${LABEL}

  # L2 shaping
  python train.py \
      --variant l2 \
      --timesteps $TIMESTEPS \
      --rollout_len $ROLLOUT \
      --run_name l2_${LABEL}

  # L2‑squared shaping
  python train.py \
      --variant l2sq \
      --timesteps $TIMESTEPS \
      --rollout_len $ROLLOUT \
      --run_name l2sq_${LABEL}


#   # Decay shaping with three horizons
# for DECAY in 200000 300000 500000; do
#     python train.py \
#             --variant decay \
#             --timesteps $TIMESTEPS \
#             --rollout_len $ROLLOUT \
#             --decay_steps $DECAY \
#             --run_name decay${DECAY}_${LABEL}
# done

