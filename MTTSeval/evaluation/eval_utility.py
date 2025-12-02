import random
import numpy as np
import torch
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.configs import get_config
from utils.model import GenHPF
from utils.train_utils import EarlyStopping
from utils.tstr_utils.data_loader import *
from utils.tstr_utils.extract_label import process_dataframes
from utils.tstr_utils.criterion import PredCriterion
from utils.convert_table_to_text import process_and_merge_events_parallel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(config, dataloaders, model, optimizer, criterion, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=config["patience"], mode="max", save_path=save_path)

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 30)
        
        # Training Phase
        model.train()
        criterion.reset()
        for batch in tqdm(dataloaders["train"], desc="Training"):
            input_ids = batch["input_ids"].to(device)
            targets = {k: v.to(device) for k, v in batch["labels"].items()}
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion("loss", outputs, targets)
            loss["total_loss"].backward()
            optimizer.step()
            with torch.no_grad():
                criterion("acc", outputs, targets)

        train_epoch_log = criterion.get_epoch_dict(len(dataloaders["train"]))
        print(f"Training Results: {train_epoch_log}")
        
        # Validation Phase
        model.eval()
        criterion.reset()
        with torch.no_grad():
            for batch in tqdm(dataloaders["valid"], desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                targets = {k: v.to(device) for k, v in batch["labels"].items()}
                
                outputs = model(input_ids)
                criterion("acc", outputs, targets)

        valid_epoch_log = criterion.get_epoch_dict(len(dataloaders["valid"]))
        print(f"Validation Results: {valid_epoch_log}")

        # Check Early Stopping
        current_metric = valid_epoch_log[criterion.update_target]
        early_stopping(current_metric, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("Training complete.")


def test_model(config, dataloader, model, criterion, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load the saved model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    print(f"Loaded model from {model_path}")

    # Reset criterion to collect metrics
    criterion.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            targets = {k: v.to(device) for k, v in batch["labels"].items()}

            # Forward pass
            outputs = model(input_ids)
            criterion("acc", outputs, targets)

    # Get test results
    test_results = criterion.get_epoch_dict(len(dataloader))
    print(f"Test Results: {test_results}")
    return test_results


def run_tstr(config):
    # Load tables and preprocess
    ehr = config["ehr"]
    obs_size = config["obs_size"]
    table_names = config["table_names"]
    real_data_root = config["real_data_root"]
    syn_data_root = config["syn_data_root"]
    pid_col = config['pid_column']
    split_col = config['split_column']

    # Define output directory
    output_data_root = config["output_data_root"]
    print("Save at ", output_data_root)
    os.makedirs(output_data_root, exist_ok=True)
    model_path = os.path.join(output_data_root, f"best_pred_model_{config['seed']}.pth")

    # Set random seed
    set_seed(config["seed"])

    real_df, syn_df = load_tables(config)

    # Load and preprocess real split data
    original_real_split_df= pd.read_csv(os.path.join(real_data_root, config["split_file_name"]))
    original_real_split_df = original_real_split_df.reset_index().rename(columns={"index": pid_col})

    real_splits = process_dataframes(real_df, config, stay_id_df=original_real_split_df[[pid_col, split_col]])
    
    total_counts = real_splits[split_col].isin(["train", "valid"]).sum()
    syn_stay_ids = set()
    for k, v in syn_df.items():
        syn_stay_ids.update(v[pid_col].unique())
    
    if len(syn_stay_ids) > total_counts:
        sorted_stay_ids = sorted(syn_stay_ids)
        kept_stay_ids = set(sorted_stay_ids[:total_counts])
        for k in syn_df.keys():
            syn_df[k] = syn_df[k][syn_df[k][pid_col].isin(kept_stay_ids)]

    syn_splits = process_dataframes(syn_df, config, stay_id_df=None) if syn_data_root != real_data_root else real_splits.copy()
    # Prevent label leakage: filter data to use observations up to half of the observation window
    for table_name in table_names:
        real_df[table_name] = real_df[table_name][real_df[table_name][config['time_column']] <= (obs_size // 2) * 60]
        syn_df[table_name] = syn_df[table_name][syn_df[table_name][config['time_column']] <= (obs_size // 2) * 60]

    # Process text data
    if syn_data_root == real_data_root:
        real_text_df = process_and_merge_events_parallel(real_df, config, 'real')
        syn_text_df = real_text_df.copy()
    else:
        # Filter real test splits
        real_test_splits = real_splits[real_splits[split_col] == "test"]
        for table_name in table_names:
            real_df[table_name] = real_df[table_name][real_df[table_name][pid_col].isin(real_test_splits[pid_col].unique())]
            
        # Process real and synthetic text data
        real_text_df = process_and_merge_events_parallel(real_df, config, 'real')
        syn_text_df = process_and_merge_events_parallel(syn_df, config, 'syn')

        print(f"Shape of real_text_df after processing: {real_text_df.shape}")
        print(f"Shape of syn_text_df after processing: {syn_text_df.shape}")

        # Adjust IDs and concatenate real and synthetic data
        real_test_splits[pid_col] = real_test_splits[pid_col] + syn_splits[pid_col].max() + 1
        real_text_df[pid_col] = real_text_df[pid_col] + syn_splits[pid_col].max() + 1
        
        syn_text_df = pd.concat([syn_text_df, real_text_df], ignore_index=True)
        syn_splits = pd.concat([syn_splits, real_test_splits[syn_splits.columns]], ignore_index=True)
        
        print(f"Final shape of syn_text_df: {syn_text_df.shape}")
        print(f"Final shape of syn_splits: {syn_splits.shape}")
    
    # Merge synthetic text data with splits
    merged_df = pd.merge(syn_text_df, syn_splits, on=pid_col, how="left")
    
    # Tokenize and prepare datasets
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Create datasets and dataloaders
    datasets = create_datasets(config, merged_df, tokenizer)
    dataloaders = create_dataloaders(config, datasets, batch_size=config["batch_size"])

    # Initialize GenHPF
    model = GenHPF(config)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = PredCriterion(config)

    # Train the model
    train_model(
        config=config,
        dataloaders=dataloaders,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        save_path=model_path
    )

    test_results = test_model(
        config=config,
        dataloader=dataloaders["test"],
        model=model,
        criterion=criterion,
        model_path=model_path
    )

    with open(os.path.join(output_data_root, f"test_results_{config['seed']}.json"), "w") as json_file:
        json.dump(test_results, json_file, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ehr', type=str, required=True, choices=['mimiciv', 'eicu'], help='EHR dataset')
    parser.add_argument('--obs_size', type=int, default=12)
    parser.add_argument('--real_data_root', type=str, required=True)
    parser.add_argument('--syn_data_root', type=str, required=True)
    parser.add_argument('--output_data_root', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    config = get_config(
        ehr=args.ehr,
        obs_size=args.obs_size,
        real_data_root=args.real_data_root,
        syn_data_root=args.syn_data_root,
        output_data_root=args.output_data_root,
        seed=args.seed,
    )
    # Example: run_tstr(config)