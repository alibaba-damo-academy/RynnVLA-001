

LAUNCHER="accelerate launch \
    --num_machines 1 \
    --num_processes 8"

$LAUNCHER train.py \
    --config_file './configs/actionvae/actionvae_lerobot.yml' \
    --exp_dir './experiments/actionvae'
