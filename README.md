# Generating Multi-Table Time Series EHR from Latent Space

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.06996)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/eunbyeol-cho/RawMed)

**RawMed** is the first framework to synthesize multi-table, time-series Electronic Health Records (EHR) data that closely resembles raw EHRs. Using text-based representation and compression techniques, RawMed captures complex structures and temporal dynamics with minimal preprocessing.

## 🎯 Key Features

- **Multi-table time-series generation**: Synthesizes raw EHR data across multiple tables
- **Minimal preprocessing**: Uses text-based representation to preserve complex structures
- **Comprehensive evaluation**: Assesses distributional similarity, inter-table relationships, temporal dynamics, and privacy

## 🚀 Quick Start

### Installation

```bash
# Create conda environment
conda create -n rawmed python=3.9
conda activate rawmed

# Install dependencies
pip install sacred==0.8.5
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm pandas scikit-learn transformers wandb einops dython rapidfuzz xgboost h5py
```

### Data Preprocessing

Preprocess your EHR dataset using the integrated pipeline (see `integrated-ehr-pipeline/README.md`), then convert to NumPy format:

```bash
python ehrsyn/datamodules/preprocess.py
```

### Training

Train the model in two stages. Both scripts take the following arguments:

```
bash run_scripts/<script>.sh [dataset] [obs_window] [cuda_device] [data_root] [ckpt_root] [syn_data_root]
```

| Argument | Description | Example |
|---|---|---|
| `dataset` | Dataset name (`eicu` or `mimiciv`) | `eicu` |
| `obs_window` | Observation window in hours (`6`, `12`, or `24`) | `12` |
| `cuda_device` | Comma-separated GPU IDs | `0` or `0,1,2,3` |
| `data_root` | Path to preprocessed NumPy data | `/path/to/data` |
| `ckpt_root` | Path to save/load model checkpoints | `/path/to/ckpts` |
| `syn_data_root` | Path to save generated synthetic data | `/path/to/output` |

**Stage 1: Train RQ-VAE for event compression**
```bash
bash run_scripts/train_RQVAE.sh eicu 12 0 /path/to/data /path/to/ckpts /path/to/output
```

**Stage 2: Train AR model and sample synthetic data**

Training and sampling are separated within the script. Uncomment the training block to train, then run sampling:

```bash
bash run_scripts/train_AR.sh eicu 12 0,1,2,3 /path/to/data /path/to/ckpts /path/to/output
```

When multiple GPUs are specified (e.g., `0,1,2,3`), sampling is automatically parallelized across GPUs via `run_scripts/parallel_sample.py`. Each GPU runs an independent process with a different random seed, and results are concatenated.

### Evaluation

Evaluate the generated synthetic data using **MTTSeval**, a comprehensive evaluation framework for multi-table time-series EHRs. The evaluation pipeline consists of two steps:

1. **Postprocess**: Convert generated NumPy arrays to table format (CSV)
2. **Evaluate**: Run evaluation metrics on the postprocessed tables

```bash
cd MTTSeval
bash run_postprocess.sh [dataset] [obs_window] [cuda_device] [mode] [data_root] [syn_data_root]
bash run_eval.sh [dataset] [obs_window] [cuda_device] [data_root] [syn_data_root]
```

MTTSeval provides various evaluation metrics including:

- **Statistical distribution analysis**: Compares distributional similarity between real and synthetic data
- **Correlation analysis**: Evaluates inter-table relationships and correlations
- **Temporal dynamics analysis**: Assesses time-series patterns and temporal consistency
- **Utility evaluation**: Includes TSTR (Train on Synthetic, Test on Real) and prediction similarity metrics

See `MTTSeval/README.md` for detailed setup and usage instructions.

## 📖 Citation

If you use RawMed in your research, please cite:

```bibtex
@article{cho2025rawmed,
  title={Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing},
  author={Cho, Eunbyeol and Kim, Jiyoun and Lee, Minjae and Park, Sungjin and Choi, Edward},
  journal={arXiv preprint arXiv:2507.06996},
  year={2025}
}
```
