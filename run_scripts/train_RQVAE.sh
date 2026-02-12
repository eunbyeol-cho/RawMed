#!/bin/bash
# Usage: bash train_RQVAE.sh [dataset] [obs_window] [cuda_device] [data_root] [ckpt_root] [syn_data_root]

dataset="${1:-eicu}"
obs_window="${2:-12}"
cuda_device="${3:-0}"
data_root="${4:?'data_root is required'}"
ckpt_root="${5:?'ckpt_root is required'}"
syn_data_root="${6:?'syn_data_root is required'}"

case "${dataset}-${obs_window}" in
    "mimiciv-6")  max_event_size=165; input_index_size=2216; time_len=2 ;;
    "mimiciv-12") max_event_size=243; input_index_size=2328; time_len=2 ;;
    "mimiciv-24") max_event_size=366; input_index_size=2386; time_len=3 ;;
    "eicu-6")     max_event_size=79;  input_index_size=1328; time_len=2 ;;
    "eicu-12")    max_event_size=114; input_index_size=1369; time_len=2 ;;
    "eicu-24")    max_event_size=179; input_index_size=1389; time_len=3 ;;
    *) echo "Invalid dataset (${dataset}) or obs_window (${obs_window})"; exit 1 ;;
esac

OMP_NUM_THREADS=8 \
NUMEXPR_MAX_THREADS=128 \
CUDA_VISIBLE_DEVICES=${cuda_device} \
    python main.py with task_train_RQVAE_indep \
    input_index_size=${input_index_size} \
    max_event_size=${max_event_size} \
    time_len=${time_len} \
    obs_size=${obs_window} \
    real_input_path=${data_root} \
    input_path=${data_root} \
    output_path=${ckpt_root} \
    generated_data_path=${syn_data_root} \
    ehr=${dataset} \
    debug=True

OMP_NUM_THREADS=8 \
NUMEXPR_MAX_THREADS=128 \
CUDA_VISIBLE_DEVICES=${cuda_device} \
    python main.py with task_test_RQVAE_indep \
    input_index_size=${input_index_size} \
    max_event_size=${max_event_size} \
    time_len=${time_len} \
    obs_size=${obs_window} \
    real_input_path=${data_root} \
    input_path=${data_root} \
    output_path=${ckpt_root} \
    generated_data_path=${syn_data_root} \
    ehr=${dataset} \
    debug=True \
    test_subsets="total"
