#!/bin/bash
# Usage: bash preprocess.sh [dataset] [obs_window] [data_root]

dataset="${1:-mimiciv}"
obs_window="${2:-12}"
data_root="${3:?'data_root is required'}"

python ehrsyn/datamodules/preprocess.py \
    --data_path ${data_root}/${dataset}.h5 \
    --dataset $dataset \
    --obs_size $obs_window
