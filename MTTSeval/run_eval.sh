#!/bin/bash
# Usage: bash run_eval.sh [dataset] [obs_window] [cuda_device] [data_root] [syn_data_root]
cd MTTSeval

dataset="${1:-eicu}"
obs_window="${2:-12}"
cuda_device="${3:-0}"
data_root="${4:?'data_root is required'}"
syn_data_root="${5:?'syn_data_root is required'}"

for seed in 0; do
    OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
        --ehr $dataset \
        --obs_size $obs_window \
        --real_data_root $data_root \
        --syn_data_root $syn_data_root \
        --output_data_root results/${dataset} \
        --seed $seed
done
