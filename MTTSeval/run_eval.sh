#!/bin/bash

for SEED in 0; do
    OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python main.py \
        --ehr $EHR \
        --obs_size $OBS_SIZE \
        --real_data_root $REAL_DATA_ROOT \
        --syn_data_root $SYN_DATA_ROOT \
        --output_data_root $OUTPUT_DATA_ROOT \
        --seed $SEED
    done
