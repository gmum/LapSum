#!/bin/bash

gpuID=${1-0}

# Path configuration
dataset="?"  # Path to your dataset directory (e.g., "/data/")
dir_results="?"  # Path to save results (e.g., "./results/experiment_1")

export CUDA_VISIBLE_DEVICES=${gpuID}
mkdir -p "${dir_results}"

export PYTHONPATH=\$PYTHONPATH:../lapsum

train_params=(
   "--alpha 1.0 --lr 1.e-3 --n 3"
   "--alpha 1.0 --lr 1.e-3 --n 5"
   "--alpha 1.0 --lr 1.e-3 --n 7"
   "--alpha 1.0 --lr 1.e-3 --n 9"
   "--alpha 1.0 --lr 1.e-3 --n 15"
)

num_epochs=200

additional_opts=""
#additional_opts="--noise_prob 0.7 --noise_scale 1 --noise_type gumbel"

current_date=$(date +"%Y-%m-%d")

for it in {1..3}; do
  for ((i=0; i<${#train_params[@]}; i++)); do
    timestamp=$(date +"%Y-%m-%d_%H%M%S.%N")
    python ./soft-permutation/run_permute.py \
              --root_data ${dataset} \
              --root_save ${dir_results}/${current_date}/${timestamp} \
              --M 20 \
              --l 4 \
              --num_epochs ${num_epochs} \
              --workers 4 \
              --seed 42 \
              ${train_params[$i]} \
              ${additional_opts}

  done
done
