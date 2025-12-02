#!/bin/bash

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/convert_text_to_table.py with \
    ehr=$EHR \
    syn_data_root=$SYN_DATA_ROOT \
    real_data_root=$REAL_DATA_ROOT

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/postprocess_table_wrong.py \
    --ehr $EHR \
    --syn_data_root $SYN_DATA_ROOT \
    --real_data_root $REAL_DATA_ROOT \
    --output_data_root $OUTPUT_DATA_ROOT
