#!/bin/bash
# Usage: bash run_preprocess.sh [dataset] [obs_window] [cuda_device] [mode] [data_root] [syn_data_root]
cd MTTSeval
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

dataset="${1:-mimiciv}"
obs_window="${2:-12}"
cuda_device="${3:-0}"
mode="${4:-real}"  # "real" or "syn"
data_root="${5:?'data_root is required'}"

if [ "$mode" = "real" ]; then
    syn_data_root="${6:-${data_root}}"
else
    syn_data_root="${6:?'syn_data_root is required for syn mode'}"
fi

# Step 1: Convert text to table format
CUDA_VISIBLE_DEVICES=$cuda_device python utils/convert_text_to_table.py \
    --ehr $dataset \
    --syn_data_root $syn_data_root \
    --real_data_root $data_root

# Step 2: Postprocess tables (syn mode only)
# CUDA_VISIBLE_DEVICES=$cuda_device python utils/postprocess_table.py \
#     --ehr $dataset \
#     --syn_data_root $syn_data_root \
#     --real_data_root $data_root \
#     --output_data_root results/${dataset}
