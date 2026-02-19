# MTTSeval

Multi-table Time-series EHR evaluation framework. Provides comprehensive metrics for evaluating synthetic EHR data including distributional similarity, inter-table relationships, temporal dynamics, and utility.

## Setup

```bash
cd MTTSeval

# Create and activate conda environment
conda create -n ehrsyn python=3.9
conda activate ehrsyn

# Install the package in development mode
pip install -e .
```

## Usage

### Step 1: Postprocess

Converts text-format data to per-table CSVs, then postprocesses them (e.g., handling invalid values, aligning schemas). Both real and synthetic data need this step.

```bash
# Postprocess real data
bash run_postprocess.sh [dataset] [obs_window] [cuda_device] real [data_root]

# Postprocess synthetic data
bash run_postprocess.sh [dataset] [obs_window] [cuda_device] syn [data_root] [syn_data_root]
```

### Step 2: Run evaluation

```bash
bash run_eval.sh [dataset] [obs_window] [cuda_device] [data_root] [syn_data_root]
```

### Arguments

| Argument | Description | Example |
|---|---|---|
| `dataset` | Dataset name (`eicu` or `mimiciv`) | `eicu` |
| `obs_window` | Observation window in hours | `12` |
| `cuda_device` | GPU device ID | `0` |
| `mode` | `real` or `syn` (postprocess only) | `syn` |
| `data_root` | Path to preprocessed real data | `/path/to/data` |
| `syn_data_root` | Path to synthetic data (required for `syn` mode) | `/path/to/syn` |
