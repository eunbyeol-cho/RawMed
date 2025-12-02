#!/bin/bash

# Define the arrays for ehr and obs_size
ehr_array=("mimiciv eicu")
obs_array=(6 12 24)

# Loop over each combination of ehr and obs_size
for ehr in "${ehr_array[@]}"; do
  for obs in "${obs_array[@]}"; do
    # Set the data folder based on the ehr value
    if [ "$ehr" == "mimiciv" ]; then
      data_folder="MIMIC-IV-2.0"
    elif [ "$ehr" == "eicu" ]; then
      data_folder="eicu-2.0"
    else
      echo "Unknown EHR: $ehr"
      exit 1
    fi

    echo "Preprocessing: $ehr (obs size = $obs)"

    # Execute the Python script with the current parameters
    python main.py \
      --ehr ${ehr} \
      --data /XXX/${data_folder} \
      --dest /YYY/${ehr}-3t-${obs}h \
      --num_threads 32 \
      --readmission \
      --diagnosis \
      --seed "0,1,2" \
      --first_icu \
      --mortality \
      --long_term_mortality \
      --max_event_size 2048 \
      --max_patient_token_len 262144 \
      --obs_size ${obs} \
      --pred_size 24

  done
done