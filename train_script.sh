# Please DO NOT RUN this shell script. This is just a way for me to record down all scripts I use #

# For Training 'pilot' episodes (initial tests)
python train.py --variant none  --timesteps 100000 --run_name pilot_none
python train.py --variant l2    --timesteps 100000 --run_name pilot_l2
python train.py --variant l2sq  --timesteps 100000 --run_name pilot_l2sq
python train.py --variant decay --timesteps 100000 --decay_steps 300000 --run_name pilot_decay

# Generate results_*.csv
python eval_returns.py