# RawMed: Generating Multi-Table Time Series EHR from Latent Space

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
pip install tqdm pandas scikit-learn transformers wandb einops dython
```

### Data Preprocessing

Preprocess your EHR dataset using the integrated pipeline (see `integrated-ehr-pipeline/README.md`), then convert to NumPy format:

```bash
python ehrsyn/datamodules/preprocess.py
```

### Training

Train the model in two stages:

**Stage 1: Train RQ-VAE for event compression**
```bash
bash run_scripts/train_RQVAE.sh
```

**Stage 2: Train autoregressive model for temporal modeling**
```bash
bash run_scripts/train_AR.sh
```

### Evaluation

Evaluate the generated synthetic data using **MTTSeval**, a comprehensive evaluation framework for multi-table time-series EHRs. MTTSeval provides various evaluation metrics including:

- **Statistical distribution analysis**: Compares distributional similarity between real and synthetic data
- **Correlation analysis**: Evaluates inter-table relationships and correlations
- **Temporal dynamics analysis**: Assesses time-series patterns and temporal consistency
- **Utility evaluation**: Includes TSTR (Train on Synthetic, Test on Real) and prediction similarity metrics

To run evaluation:

```bash
cd MTTSeval
bash run_eval.sh
```

See `MTTSeval/README.md` for detailed setup and usage instructions.

## 📖 Citation

If you use RawMed in your research, please cite:

```bibtex
@article{cho2024rawmed,
  title={Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing},
  author={Cho, Eunbyeol and Kim, Jiyoun and Lee, Minjae and Park, Sungjin and Choi, Edward},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
