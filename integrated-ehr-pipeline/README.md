# Integrated-EHR-Pipeline

This repository is a cloned and lightly adapted version of the original Integrated-ehr-pipeline hosted on GitHub. It accompanies the UniHPF study and is provided here in an academic context to facilitate reproducibility and downstream experimentation within my computing environment.

- Original project: `https://github.com/Jwoo5/integrated-ehr-pipeline`
- Pre-processing code refining project in [UniHPF](https://arxiv.org/abs/2207.09858)


## Install Requirements
- NOTE: This repository requires `python>=3.9` and `Java>=8`
- NOTE: Since there is a performance issue related to `transformers` library, it is recommended to use 
`transformers==4.29.1`.

Install the Python dependencies:
```
pip install numpy pandas tqdm treelib transformers==4.29.1 pyspark
```

## Usage


```
python main.py \
  --ehr ${ehr} \
  --data ${data_folder} \
  --dest ${dest_folder} \
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
```

- The pipeline can automatically download the required datasets from PhysioNet; valid credential/certification is required.
- To use an already-downloaded dataset, specify `--data {path_to_data}`.
- A sample PyTorch `Dataset` implementation is provided in `sample_dataset.py`.


## Acknowledgments

All credit for the methods and original implementation belongs to the maintainers of the integrated-ehr-pipeline. This repository merely provides minor adaptations necessary for execution in my environment.
