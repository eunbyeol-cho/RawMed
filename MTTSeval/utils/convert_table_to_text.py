import os
import re
import pandas as pd
import numpy as np
import argparse, math
from itertools import chain
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.configs import get_config
from transformers import AutoTokenizer
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

def preprocess_table(df, stay_id_col, time_col, table_name):
    """
    Add a text column for each table and retain only the necessary columns.
    """
    # Define columns to exclude
    exclude_cols = {stay_id_col, time_col}
    include_cols = [col for col in df.columns if col not in exclude_cols]

    # Create a text column
    df['text'] = df[include_cols].apply(
        lambda row: f"{table_name} " + " ".join([f"{col} {row[col]}" for col in include_cols if pd.notnull(row[col])]),
        axis=1
    )

    # Add spaces around all numeric characters and periods (e.g., "123.45" -> " 1 2 3 . 4 5 ")
    df['text'] = df['text'].apply(lambda text: re.sub(r"([0-9\.])", r" \1 ", text))

    # Retain only necessary columns
    return df[[stay_id_col, time_col, 'text']]

def preprocess_table_chunk(table_name, df_chunk, stay_id_col, time_col):
    return preprocess_table(df_chunk, stay_id_col, time_col, table_name)

def tokenize_df(df, chunk_size=10000):
    texts = df['text'].tolist()
    num_texts = len(texts)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Calculate how many chunks we'll have
    num_chunks = math.ceil(num_texts / chunk_size)

    all_ids = []
    # Loop through each chunk
    for i in tqdm(range(num_chunks), desc="Tokenizing in chunks"):
        start = i * chunk_size
        end = (i+1) * chunk_size
        chunk_texts = texts[start:end]

        # Tokenize this batch in one call
        encoded = tokenizer(
            chunk_texts,
            max_length=128,
            truncation=True,
            padding='max_length',
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        all_ids.extend(encoded['input_ids'])

    df = df.copy()
    df['tokenized_text'] = all_ids
    return df

def process_and_merge_events_parallel(real_df, config, df_type, stay_id_col=None, time_col=None, num_workers=32, chunk_size=500000):
    if stay_id_col is None:
        stay_id_col = config["pid_column"]
    if time_col is None:
        time_col = config["time_column"]
    all_events = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for table_name, df in real_df.items():
            
            num_chunks = (len(df) // chunk_size) + 1
            for i in range(num_chunks):
                df_chunk = df.iloc[i*chunk_size : (i+1)*chunk_size]
                futures.append(executor.submit(preprocess_table_chunk, table_name, df_chunk, stay_id_col, time_col))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Chunks"):
            processed_chunk = future.result()
            all_events.append(processed_chunk)
    
    merged_events = pd.concat(all_events, ignore_index=True)
    merged_events.sort_values(by=[stay_id_col, time_col], inplace=True)
    
    print(f"{df_type} BEFORE...")
    print(merged_events)
    
    grouped_texts = merged_events.groupby(stay_id_col)['text'].apply('[SEP]'.join).reset_index()
    grouped_texts.columns = [stay_id_col, 'event_sequence']

    return grouped_texts

def process_and_merge_events(real_df, config, df_type, stay_id_col=None, time_col=None):
    if stay_id_col is None:
        stay_id_col = config["pid_column"]
    if time_col is None:
        time_col = config["time_column"]
    """
    Process and merge all events and sort them chronologically.
    """
    all_events = []

    # Process each table
    for table_name, df in tqdm(real_df.items(), desc="Processing Tables"):
        processed_df = preprocess_table(df, stay_id_col, time_col, table_name)
        all_events.append(processed_df)

    # Merge all tables
    merged_events = pd.concat(all_events)
    # Sort by time in chronological order
    merged_events.sort_values(by=[stay_id_col, time_col], inplace=True)
    
    # Group texts by stay ID
    grouped_texts = merged_events.groupby(stay_id_col)['text'].apply('[SEP]'.join).reset_index()
    grouped_texts.columns = [stay_id_col, 'event_sequence']

    return grouped_texts

def load_tables(config):
    # Load tables and preprocess
    real_data_root = config["real_data_root"]
    syn_data_root = config["syn_data_root"]
    table_names = config["table_names"]

    real_dataframes = {}
    synthetic_dataframes = {}

    for table_name in table_names:
        # Load data
        real_df = pd.read_csv(os.path.join(real_data_root, f"{table_name}.csv"))
        synthetic_df = pd.read_csv(os.path.join(syn_data_root, f"{table_name}.csv"))

        synthetic_df = synthetic_df[real_df.columns]
        
        if config["sample"]:
            real_df = real_df.iloc[:10000]
            synthetic_df = synthetic_df.iloc[:10000]
        
        # Store in dictionaries
        real_dataframes[table_name] = real_df.copy()
        synthetic_dataframes[table_name] = synthetic_df.copy()
    
    return real_dataframes, synthetic_dataframes
