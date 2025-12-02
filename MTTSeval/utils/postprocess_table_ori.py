import os
import argparse
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process
from utils.convert_table_to_text import load_tables
from utils.configs import get_config

def filter_columnwise_outliers(real_df, syn_df, numeric_columns, pid_col):
    """Filters out rows in syn_df where any numeric column value is outside the min-max range observed in real_df."""
    if not numeric_columns:
        return syn_df, set()
    
    # Calculate min and max for numeric columns
    min_values = real_df[numeric_columns].min()
    max_values = real_df[numeric_columns].max()

    # Identify rows with any numeric outliers
    is_outlier = syn_df[numeric_columns].apply(
        lambda col: (col < min_values[col.name]) | (col > max_values[col.name])
    )
    any_outlier = is_outlier.any(axis=1)

    # Identify and drop stay_ids with outliers
    dropped_stay_ids = set(syn_df.loc[any_outlier, pid_col])
    print(f"Step 1 - Numeric outliers removed: {len(dropped_stay_ids)} {pid_col}s")

    # Filter the DataFrame
    syn_df_filtered = syn_df.loc[~any_outlier].copy()
    return syn_df_filtered, dropped_stay_ids

def filter_columnwise_categorical_vocab(real_df, syn_df, col_type, threshold, pid_col):
    """Filters rows in syn_df with invalid categorical values or replaces them with closest matches."""
    dropped_stay_ids = set()
    categorical_columns = col_type.get("categorical_columns", [])

    for col in categorical_columns:
        # Ensure columns are strings for matching
        real_df[col] = real_df[col].astype(str)
        syn_df[col] = syn_df[col].astype(str)

        if col in ["isopenbag", "labtypeid"]:
            replacement_map = {f"{k}.0": k for k in real_df[col].unique()}
            syn_df[col] = syn_df[col].replace(replacement_map)

        valid_vocab = set(real_df[col].unique())

        # Identify invalid rows
        invalid_rows_mask = ~syn_df[col].isin(valid_vocab)
        invalid_indices = syn_df.index[invalid_rows_mask]
    
        print(" ->", col, len(invalid_indices))

        for idx in invalid_indices:
            value = syn_df.at[idx, col]
            # Attempt to find the closest match
            match = process.extractOne(value, valid_vocab, scorer=Levenshtein.normalized_distance)
            if match and match[1] <= threshold:
                syn_df.at[idx, col] = match[0]  # Replace with closest match
            else:
                dropped_stay_ids.add(syn_df.at[idx, pid_col])

    print(f"Step 2 - Categorical vocab validation removed: {len(dropped_stay_ids)} {pid_col}s")

    # Filter the DataFrame
    syn_df_filtered = syn_df[~syn_df[pid_col].isin(dropped_stay_ids)].copy()
    return syn_df_filtered, dropped_stay_ids


def filter_itemwise_outliers(real_df, syn_df, icol, numeric_columns, pid_col):
    """
    Filters rows in syn_df where numeric column values fall outside the item-wise min-max range observed in real_df.
    """
    if not numeric_columns:
        return syn_df, set()

    # Step 1: Calculate min and max for each group in the real table
    grouped_real = real_df.groupby(icol)
    real_min = grouped_real[numeric_columns].min().reset_index()
    real_max = grouped_real[numeric_columns].max().reset_index()

    # Step 2: Merge min and max values with the synthetic table
    syn_df_with_limits = syn_df.merge(real_min, on=icol, suffixes=('', '_min'))
    syn_df_with_limits = syn_df_with_limits.merge(real_max, on=icol, suffixes=('', '_max'))

    # Step 3: Identify rows to discard
    out_of_bounds_mask = False  # Initialize a boolean mask
    for col in numeric_columns:
        if col == "patientweight":
            print(f"'patientweight' is not related to '{icol}'")
            continue
        out_of_bounds_mask |= (syn_df_with_limits[col] < syn_df_with_limits[f"{col}_min"]) | \
                              (syn_df_with_limits[col] > syn_df_with_limits[f"{col}_max"])

    discarded_rows = syn_df_with_limits[out_of_bounds_mask]

    # Step 4: Collect unique `stay_id` values from discarded rows
    dropped_ids = discarded_rows['stay_id'].unique()

    # Step 5: Remove rows with invalid values and cleanup temporary columns
    syn_df_filtered = syn_df_with_limits[
        ~syn_df_with_limits['stay_id'].isin(dropped_ids)
    ].drop(columns=[f"{col}_min" for col in numeric_columns] + [f"{col}_max" for col in numeric_columns])

    print(f"Step 3 - {icol}-wise numeric outliers removed: {len(dropped_ids)} stay_ids")

    return syn_df_filtered, dropped_ids

def filter_itemwise_categorical_vocab(real_df, syn_df, icol, col_type, threshold, pid_col):
    """
    Filters rows in syn_df with invalid item-wise categorical values or replaces them with the closest matches based on real_df vocabularies.
    """
    dropped_stay_ids = set()
    categorical_columns = col_type.get("categorical_columns", [])
    categorical_columns.remove(icol)
    
    if not categorical_columns:
        print("No categorical columns to process.")
        return syn_df, set()

    # Group valid vocabularies for each item
    item_vocab = real_df.groupby(icol).agg(
        {col: lambda x: set(x.astype(str).unique()) for col in categorical_columns}
    ).reset_index()
    
    # Merge item vocabularies into synthetic data
    syn_df = syn_df.merge(item_vocab, on=icol, suffixes=('', '_valid'), how='left')

    # Prepare list to track invalid rows
    invalid_rows = []

    for col in categorical_columns:
        # Get valid vocab for each row
        syn_df[f'{col}_valid'] = syn_df[f'{col}_valid'].apply(lambda x: x if isinstance(x, set) else set())
        
        # Identify rows with invalid values
        invalid_mask = ~syn_df.apply(lambda row: row[col] in row[f'{col}_valid'], axis=1)
        invalid_indices = syn_df.index[invalid_mask]

        # Attempt to replace invalid values with closest match
        for idx in invalid_indices:
            value = syn_df.at[idx, col]
            valid_vocab = syn_df.at[idx, f'{col}_valid']
            if valid_vocab:
                match = process.extractOne(value, valid_vocab, scorer=Levenshtein.normalized_distance)
                if match and match[1] <= threshold:
                    syn_df.at[idx, col] = match[0]  # Replace with closest match
                else:
                    invalid_rows.append(idx)  # Mark for removal
                    dropped_stay_ids.add(syn_df.at[idx, pid_col])
            else:
                invalid_rows.append(idx)  # Mark for removal
                dropped_stay_ids.add(syn_df.at[idx, pid_col])

        print(f" -> {col}: Invalid rows processed = {len(invalid_rows)}")

    # Drop invalid rows
    syn_df_filtered = syn_df[~syn_df.index.isin(invalid_rows)].copy()

    # Drop temporary columns
    syn_df_filtered.drop(columns=[f'{col}_valid' for col in categorical_columns], inplace=True)

    print(f"Step 4 - {icol}-wise categorical vocab validation removed: {len(dropped_stay_ids)} stay_ids")
    
    return syn_df_filtered, dropped_stay_ids

def extract_itemid_col(task_config):
    itemid_col = {}
    for task, table_config in task_config.items():
        itemid_col[table_config["table_name"]] = table_config["itemid_col"]
    return itemid_col

def main(config, cut=False, cut_samples=None):
    # Load tables and preprocess
    ehr = config["ehr"]
    real_data_root = config["real_data_root"]
    syn_data_root = config["syn_data_root"]
    output_data_root = config["output_data_root"]
    
    threshold = config["threshold"]
    postprocess_steps = config["postprocess_steps"]
    itemid_column = extract_itemid_col(config["task_config"])
    split_col = config['split_column']
    pid_col = config["pid_column"]
    time_col = config["time_column"]
    
    real_dfs, syn_dfs = load_tables(config)

    col_type = pd.read_pickle(os.path.join(real_data_root, config["col_type"]))
    splits = pd.read_csv(os.path.join(real_data_root, config["split_file_name"])).reset_index()
    train_indices = splits[splits[split_col] == "train"]["index"]

    # Initialize variables to track dropped stay_ids and outputs
    all_dropped_stay_ids = set()
    intermediate_outputs = {}

    os.makedirs(output_data_root, exist_ok=True)

    for table_name in config["table_names"]:
        # if table_name in ["labevents", "prescriptions", "medication", "lab"]:
        #     continue
        print(f"\nProcessing table: {table_name}")

        real_df = real_dfs[table_name]
        real_df = real_df[real_df[pid_col].isin(train_indices)]
        syn_df = syn_dfs[table_name]
        
        numeric_columns = col_type[table_name]["numeric_columns"]

        # Step 1: Filter columnwise numeric outliers
        if 1 in postprocess_steps:
            syn_df, numeric_dropped_ids = filter_columnwise_outliers(real_df, syn_df, numeric_columns, pid_col)
            all_dropped_stay_ids.update(numeric_dropped_ids)

        # Step 2: Filter columnwise categorical vocab
        if 2 in postprocess_steps:
            syn_df, vocab_dropped_ids = filter_columnwise_categorical_vocab(real_df, syn_df, col_type[table_name], threshold, pid_col)
            all_dropped_stay_ids.update(vocab_dropped_ids)

        # Step 3: Filter itemwise numeric outliers
        if 3 in postprocess_steps:
            icol = itemid_column[table_name]
            if icol:
                syn_df, itemwise_numeric_dropped_ids = filter_itemwise_outliers(real_df, syn_df, icol, numeric_columns, pid_col)
                all_dropped_stay_ids.update(itemwise_numeric_dropped_ids)
        
        # Step 4: Filter itemwise categorical vocab
        if 4 in postprocess_steps:
            icol = itemid_column[table_name]
            if icol:
                syn_df, itemwise_categorical_dropped_ids = filter_itemwise_categorical_vocab(real_df, syn_df, icol, col_type[table_name], threshold, pid_col)
                all_dropped_stay_ids.update(itemwise_categorical_dropped_ids)

        # Store intermediate output
        intermediate_outputs[table_name] = syn_df

    # Remove globally invalid stay_ids across all tables
    for table_name, syn_df in intermediate_outputs.items():
        final_syn_df = syn_df[~syn_df[pid_col].isin(all_dropped_stay_ids)]
        print(f"Final number of rows for '{table_name}': {len(final_syn_df)}")

        final_syn_df.to_csv(os.path.join(output_data_root, f"{table_name}.csv"), index=False)
    print(f"Total unique {pid_col}s removed across all tables: {len(all_dropped_stay_ids)}")

    remained_stay_ids = set()
    for table_name in config["table_names"]:
        final_syn_df = pd.read_csv(os.path.join(output_data_root, f"{table_name}.csv"))
        remained_stay_ids.update(final_syn_df[pid_col].unique())
    print(f"Total unique {pid_col}s remaining across all tables: {len(remained_stay_ids)}")
    
    if cut and cut_samples is not None:
        if cut_samples > len(remained_stay_ids):
            raise ValueError(f"Requested {cut_samples} {pid_col}s, but only {len(remained_stay_ids)} are available.")
        
        sampled_stay_ids = list(remained_stay_ids)[:cut_samples]
        print(f"Sampled {len(sampled_stay_ids)} {pid_col}s for cut mode.")

        #  Filter each table to include only sampled stay_ids
        for table_name in config["table_names"]:
            
            final_syn_df = pd.read_csv(os.path.join(output_data_root, f"{table_name}.csv"))
            filtered_df = final_syn_df[final_syn_df[pid_col].isin(sampled_stay_ids)]
            filtered_df.to_csv(os.path.join(output_data_root, f"{table_name}.csv"), index=False)
            print(f"Saved filtered '{table_name}' with {len(filtered_df)} rows.")

        # Update remained_stay_ids after sampling
        remained_stay_ids = sampled_stay_ids
        print(f"Total unique {pid_col}s after sampling: {len(remained_stay_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ehr', type=str, required=True, choices=['mimiciv', 'eicu'], help='EHR dataset')
    parser.add_argument('--obs_size', type=int, default=12)
    parser.add_argument('--real_data_root', type=str, required=True)
    parser.add_argument('--syn_data_root', type=str, required=True)
    parser.add_argument('--output_data_root', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cut', action='store_true')
    parser.add_argument('--cut_samples', type=int, default=None)
    args = parser.parse_args()

    config = get_config(
        ehr=args.ehr,
        obs_size=args.obs_size,
        real_data_root=args.real_data_root,
        syn_data_root=args.syn_data_root,
        output_data_root=args.output_data_root,
        seed=args.seed,
    )
    main(config, cut=args.cut, cut_samples=args.cut_samples)