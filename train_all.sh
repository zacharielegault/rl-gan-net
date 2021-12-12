#! /bin/bash
set -e  # Exit this script if an error occurs in one of the following commands

start_time=$(date +%Y_%m_%d_%H_%M_%S)
config_path=config_${start_time}.yaml

cp default_config.yaml "${config_path}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate rl-gan-net
python ae.py --config "${config_path}" --save_checkpoint
python gan.py --config "${config_path}" --save_checkpoint
python rl_gan_net.py --config "${config_path}"
