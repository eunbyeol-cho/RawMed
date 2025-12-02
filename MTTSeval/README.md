# MTTSeval
Multi-table Time-series EHR evaluThis project provides a framework for evaluating multi-table time-series Electronic Health Records (EHRs). This README outlines the setup instructions and steps to run the evaluation scripts.

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd MTTSeval
```

2. Create and activate a conda environment:
```bash
conda create -n ehrsyn python=3.9
conda activate ehrsyn
```

3. Install the package in development mode:
```bash
pip install -e .
```

This will install the package and all its dependencies, and set up the proper Python path for importing modules.

## Running Evaluation Scripts

After completing the setup, you can run the following scripts:  

- **`run_postprocess.sh`**: Postprocesses generative tables, converting text to table format and performing additional postprocessing.  
- **`run_eval.sh`**: Executes the evaluation pipeline.  

Run the scripts directly from the terminal:  
```bash
bash run_postprocess.sh
bash run_eval.sh