dataset="mimiciv"
obs_window=12
folder_name="${dataset}-3t-${obs_window}h"

# Define max_event_size and input_index_size based on dataset and obs_window
case "${dataset}-${obs_window}" in
    "mimiciv-6")
        max_event_size=165
        input_index_size=2216
        time_len=2
        ;;
    "mimiciv-12")
        max_event_size=243
        input_index_size=2328
        time_len=2
        ;;
    "mimiciv-24")
        max_event_size=366
        input_index_size=2386
        time_len=3
        ;;
    "eicu-6")
        max_event_size=79
        input_index_size=1328
        time_len=2
        ;;
    "eicu-12")
        max_event_size=114
        input_index_size=1369
        time_len=2
        ;;
    "eicu-24")
        max_event_size=179
        input_index_size=1389
        time_len=3
        ;;
    *)
        echo "Invalid dataset (${dataset}) or obs_window (${obs_window})"
        exit 1
        ;;
esac

gpu_id=0

echo "${dataset} ${obs_window} ${max_event_size} ${input_index_size} ${time_len} "

OMP_NUM_THREADS=8 \
NUMEXPR_MAX_THREADS=128 \
CUDA_VISIBLE_DEVICES=${gpu_id} \
    python main.py with task_train_RQVAE_indep \
    input_index_size=${input_index_size} \
    max_event_size=${max_event_size} \
    time_len=${time_len} \
    obs_size=${obs_window} \
    real_input_path=${real_input_path} \
    input_path=${real_input_path} \
    output_path=${output_path} \
    generated_data_path=${generated_data_path} \
    ehr=${dataset} \
    debug=True

    
OMP_NUM_THREADS=8 \
NUMEXPR_MAX_THREADS=128 \
CUDA_VISIBLE_DEVICES=${gpu_id} \
    python main.py with task_test_RQVAE_indep \
    input_index_size=${input_index_size} \
    max_event_size=${max_event_size} \
    time_len=${time_len} \
    obs_size=${obs_window} \
    real_input_path=${real_input_path} \
    input_path=${real_input_path} \
    output_path=${output_path} \
    generated_data_path=${generated_data_path} \
    ehr=${dataset} \
    debug=True \
    test_subsets="total"


