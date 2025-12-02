import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.convert_table_to_text import *
from utils.train_utils import get_task
from concurrent.futures import ProcessPoolExecutor, as_completed

def tokenize_texts(df, tokenizer, max_len):
    def tokenize_and_pad(text):
        events = text.split("[SEP]")
        tokenized_events = [
            tokenizer.encode(event, truncation=True, max_length=max_len, add_special_tokens=False)
            for event in events
        ]

        padded_events = [
            tokens + [0] * (max_len - len(tokens)) if len(tokens) < max_len else tokens
            for tokens in tokenized_events
        ]
        return padded_events

    # df["input_ids"] = df["event_sequence"].apply(tokenize_and_pad)
    # Wrap the `apply` function with `tqdm` for progress tracking
    tqdm.pandas(desc="Tokenizing rows")
    df["input_ids"] = df["event_sequence"].progress_apply(tokenize_and_pad)

    return df

def tokenize_text_chunk(df_chunk, tokenizer, max_len):
    """
    Tokenize and pad a chunk of the DataFrame.
    """
    def tokenize_and_pad(text):
        events = text.split("[SEP]")
        tokenized_events = [
            tokenizer.encode(event, truncation=True, max_length=max_len, add_special_tokens=False)
            for event in events
        ]
        
        padded_events = [
            tokens + [0] * (max_len - len(tokens)) if len(tokens) < max_len else tokens
            for tokens in tokenized_events
        ]
        return padded_events

    df_chunk["input_ids"] = df_chunk["event_sequence"].apply(tokenize_and_pad)
    return df_chunk

def tokenize_texts_parallel(df, tokenizer, max_len, chunk_size=1000, num_workers=32):
    """
    Tokenize texts in parallel with progress tracking.
    """
    all_chunks = []
    num_chunks = (len(df) // chunk_size) + 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_chunks):
            df_chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
            futures.append(executor.submit(tokenize_text_chunk, df_chunk, tokenizer, max_len))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing Chunks"):
            processed_chunk = future.result()
            all_chunks.append(processed_chunk)

    # Combine all processed chunks into a single DataFrame
    tokenized_df = pd.concat(all_chunks, ignore_index=True)
    return tokenized_df

def process_labels(config, df):
    labels = {}
    pred_tasks = [get_task(task) for task in config["pred_tasks"]]

    for task in pred_tasks:
        task_name = task.name
        task_prop = task.property
        task_class = task.num_classes

        array = df[f"label_{task_name}"].values
        if task_prop == "binary":
            array = np.nan_to_num(array, nan=-1) 
            labels[task_name] = torch.tensor(array, dtype=torch.float32)
        elif task_prop == "multi-class":
            valid_indices = ~np.logical_or(array == -1, np.isnan(array))
            one_hot_encoded = torch.zeros((len(array), task_class), dtype=torch.float32)
            one_hot_encoded[valid_indices] = torch.eye(task_class)[array[valid_indices].astype(int)]
            labels[task_name] = one_hot_encoded

    return labels

class ClinicalTextDataset(Dataset):
    def __init__(self, config, tokenized_df):
        self.df = tokenized_df
        self.labels = process_labels(config, tokenized_df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = {key: self.labels[key][idx] for key in self.labels.keys()}
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "label": label,
        }

def create_datasets(config, df, tokenizer):
    """
    Tokenize texts and split the dataset into train/valid/test.
    """
    
    max_len = config["max_event_token_len"]
    tokenized_df = tokenize_texts_parallel(df, tokenizer, max_len)
    
    datasets = {}

    for split in ["train", "valid", "test"]:
        split_df = tokenized_df[tokenized_df[config["split_column"]] == split]
        datasets[split] = ClinicalTextDataset(config, split_df)

    return datasets

def create_dataloaders(config, datasets, batch_size):
    """
    Create DataLoader for each split.
    """
    def custom_collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["label"] for item in batch]
        max_event_size = config["max_event_size"]

        # Pad input_ids
        padded_input_ids = [
            F.pad(input_id, (0, 0, 0, max_event_size - len(input_id)), mode="constant", value=0)
            if len(input_id) < max_event_size else input_id[:max_event_size]
            for input_id in input_ids
        ]
        padded_input_ids = torch.stack(padded_input_ids)

        # Batch labels dynamically
        if isinstance(labels[0], dict):  # Multi-task case
            batched_labels = {
                task: torch.stack([label[task] for label in labels]) for task in labels[0].keys()
            }
        else:  # Single-task case
            batched_labels = torch.tensor(labels, dtype=torch.long)

        return {"input_ids": padded_input_ids, "labels": batched_labels}

    return {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=custom_collate_fn,
        )
        for split in datasets.keys()
    }