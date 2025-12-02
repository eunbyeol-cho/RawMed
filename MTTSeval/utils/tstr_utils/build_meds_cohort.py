import numpy as np
import pandas as pd
import os
import shutil
import glob
import pickle
import warnings
import argparse
from utils.configs import get_config
from convert_ehr_to_meds import convert_tables_to_meds
warnings.filterwarnings(action='ignore')

def get_last_non_null_values(df, obs_size, task_string, lab_config, config):
    """
    Extract the last non-null values for specific tasks within a specified observation size.
    """
    pid_col = config['pid_column']
    time_col = config['time_column']
    itemid_col = lab_config["itemid_col"]
    value_col = lab_config["value_col"]

    # Create a dictionary to map each item to its corresponding task
    item_to_task = {}
    for task, itemids in task_string.items():
        for itemid in itemids:
            item_to_task[itemid] = task

    # Filter by time and relevant itemids
    relevant_itemids = set(item_to_task.keys())
    filtered_df = df[(df['time'] > (obs_size // 2) * 60) & (df[itemid_col].isin(relevant_itemids))].copy()

    # Map the item IDs in the dataframe to the corresponding task
    filtered_df['task'] = filtered_df[itemid_col].map(item_to_task)

    # Sorting to ensure last value is the latest
    filtered_df = filtered_df.sort_values(by='time')

    # Aggregating last non-null values for each task per stay_id
    last_values = filtered_df.groupby(['stay_id', 'task']).agg({
        value_col: lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None
    }).reset_index()

    return last_values

def apply_binning(last_values, lab_test_bins, lab_labels, task_string, lab_config, config):
    """
    Apply binning to the last values and generate labels for each task.
    """
    pid_col = config['pid_column']
    value_col = lab_config["value_col"]
    
    # Prepare an empty DataFrame to collect results
    results = pd.DataFrame({'stay_id': last_values['stay_id'].unique()})
    
    # Apply binning for each test
    for test, bins in lab_test_bins.items():
        # Filter relevant values by task
        relevant_values = last_values[last_values['task'] == test]
        
        def get_label(row):
            if not pd.isnull(row[value_col]):
                return pd.cut([row[value_col]], bins=bins, labels=lab_labels[test])[0]
            return None
        
        # Apply binning and create a new column with labels
        relevant_values['label_' + test] = relevant_values.apply(get_label, axis=1)
        
        # Merge the results with the main DataFrame
        results = results.merge(relevant_values[['stay_id', 'label_' + test]], on='stay_id', how='left')
    
    return results

def extract_labels(df, obs_size, config):
    """
    Generate binary labels for each task within the specified observation window.
    """
    pid_col = config['pid_column']
    time_col = config['time_column']
    itemid_col = config["itemid_col"]
    itemids = config["task"]

    def process_itemid_data(df, itemid, obs_size):
        itemid = itemid.lower()
        filtered_df = df[df[itemid_col].str.contains(itemid, case=False, na=False)]
        
        # Define time cutoffs
        time_cutoff = (obs_size // 2) * 60
        next_window_cutoff = 2 * time_cutoff

        # Identify stay_ids within the following 6 hours
        stay_ids_after_6_hours = set(filtered_df[
            (filtered_df['time'] > time_cutoff) & (filtered_df['time'] <= next_window_cutoff)
        ]['stay_id'].unique())

        return stay_ids_after_6_hours

    # Prepare label data
    label_data = {'stay_id': list(df['stay_id'].unique())}
    
    for itemid in itemids:
        # Process each itemid
        stay_ids_after_6_hours = process_itemid_data(df, itemid, obs_size)
        label_data[f'label_{itemid}'] = [
            1 if stay_id in stay_ids_after_6_hours else 0 for stay_id in label_data['stay_id']
        ]
    
    # Convert to DataFrame
    labels_df = pd.DataFrame(label_data)
    return labels_df
  
def process_dataframes(dataframes, obs_size, config, ehr_key, task_map, lab_bins, labels, total_stay_id_df=None):
    """
    Processes multiple DataFrames and combines them into a single output DataFrame.
    """
    pid_col = config['pid_column']
    time_col = config['time_column']
    split_col = config['split_column']

    # Process lab data
    lab_config = config[ehr_key]["lab"]
    last_lab_values = get_last_non_null_values(
        dataframes[lab_config["table_name"]], obs_size, task_map[ehr_key], lab_config, config
    )
    lab_label_df = apply_binning(
        last_lab_values, lab_bins, labels, task_map[ehr_key], lab_config, config
    )
    print("Lab data processed.")

    # Process input data
    input_label_df = extract_labels(
        dataframes[config[ehr_key]['input']['table_name']],
        obs_size,
        config[ehr_key]['input']
    )
    print("Input data processed.")

    # Process medication data
    med_label_df = extract_labels(
        dataframes[config[ehr_key]['med']['table_name']],
        obs_size,
        config[ehr_key]['med']
    )
    print("Medication data processed.")

    # Compute row counts for stay_id from filtered DataFrame
    threshold_time = (obs_size // 2) * 60
    filtered = pd.concat([
        df[df['time'] <= threshold_time][[pid_col]]
        for df in dataframes.values()
    ])
    counts = filtered[pid_col].value_counts().reset_index()
    counts.columns = [pid_col, 'total_half_event']

    # Either create a new output_df or update the existing one
    if total_stay_id_df is None:
        # Collect all unique stay_id from dataframes
        total_stay_ids = set()
        for df in dataframes.values():
            total_stay_ids.update(df[pid_col].unique())
        total_stay_id_df = pd.DataFrame({pid_col: list(total_stay_ids)})

        # Merge counts with total_stay_id_df and handle missing values
        total_stay_id_df = total_stay_id_df.merge(counts, on=pid_col, how='left').fillna(0)
        total_stay_id_df['total_half_event'] = total_stay_id_df['total_half_event'].astype(int)
        
        split_df = total_stay_id_df \
            .merge(input_label_df, on=pid_col, how='left') \
            .merge(med_label_df, on=pid_col, how='left') \
            .merge(lab_label_df, on=pid_col, how='left')
    else:
        # Merge counts with total_stay_id_df and handle missing values
        total_stay_id_df = total_stay_id_df.merge(counts, on=pid_col, how='left').fillna(0)
        total_stay_id_df['total_half_event'] = total_stay_id_df['total_half_event'].astype(int)
        
        split_df = total_stay_id_df \
            .merge(input_label_df, on=pid_col, how='left') \
            .merge(med_label_df, on=pid_col, how='left') \
            .merge(lab_label_df, on=pid_col, how='left')
    print(split_df.shape)

    # Identify label columns and set values to None where row_count <= 5
    label_columns = [col for col in split_df.columns if 'label' in col]
    print(f"Total nullified label values: {len(split_df[split_df['total_half_event'] <= 5])}")
    split_df.loc[split_df['total_half_event'] <= 5, label_columns] = None
    
    # Return the processed DataFrame
    return split_df

def split_train_valid(df, seed=0, test_size=0.11, config=None):
    np.random.seed(seed)  # Set random seed for reproducibility
    # Shuffle the indices of the DataFrame
    df = df.copy()
    shuffled_indices = np.random.permutation(df.index)
    
    # Calculate the split point
    split_point = int(len(shuffled_indices) * (1 - test_size))
    
    # Split the indices into train and valid
    train_indices = shuffled_indices[:split_point]
    valid_indices = shuffled_indices[split_point:]
    
    # Assign "train" and "valid" to the split column
    df.loc[train_indices, config["split_column"]] = "train"
    df.loc[valid_indices, config["split_column"]] = "valid"
    
    return df

def process_and_split_data(dataframes, obs_size, config, ehr_key, task_map, lab_bins, labels, stay_id_df=None):
    """Processes and splits dataframes for synthetic and real datasets."""
    split_df = process_dataframes(
        dataframes=dataframes,
        obs_size=obs_size,
        config=config,
        ehr_key=ehr_key,
        task_map=task_map,
        lab_bins=lab_bins,
        labels=labels,
        total_stay_id_df=stay_id_df
    )
    if stay_id_df is None:
        split_df = split_train_valid(split_df.copy(), config=config)
    return split_df

def save_split_data(root_path, split_df, seed_column, meds_data, split_map, is_real=True, config=None):
    """Saves split data (train, valid, test) into appropriate folders."""
    pid_col = config['pid_column']
    for split_name, output_folder in split_map.items():
        os.makedirs(os.path.join(root_path, "data", output_folder), exist_ok=True)
        split_stay_ids = split_df[split_df[seed_column] == split_name][pid_col].unique()
        
        if is_real:
            filtered_data = meds_data[meds_data.subject_id.isin(split_stay_ids)]
        else:
            # Synthetic 데이터의 경우, test split은 제외
            if split_name == "test":
                continue
            filtered_data = meds_data[meds_data.subject_id.isin(split_stay_ids)]
        
        output_path = os.path.join(root_path, "data", output_folder, "0.parquet")
        filtered_data.to_parquet(output_path)
        
        print(f"{'Real' if is_real else 'Synthetic'} {split_name}: {len(split_stay_ids)} stays, "
              f"Shape: {filtered_data.shape}, Path: {output_path}")

def process_test_split(root_path, split_df, seed_column, meds_data, new_id_start, config):
    """Processes and saves test split with remapped subject IDs."""
    pid_col = config['pid_column']
    test_stay_ids = split_df[split_df[seed_column] == "test"][pid_col].unique()
    new_stay_ids = {old_id: new_id for new_id, old_id in enumerate(test_stay_ids, start=new_id_start)}

    test_data = meds_data[meds_data.subject_id.isin(test_stay_ids)].copy()
    test_data["subject_id"] = test_data["subject_id"].map(new_stay_ids)

    output_path = os.path.join(root_path, "data", "held_out", "0.parquet")
    test_data.to_parquet(output_path)

    print(f"Test split remapped: {len(test_stay_ids)} stays, Shape: {test_data.shape}, Path: {output_path}")
    return test_stay_ids, new_stay_ids

def analyze_label_counts(real_split_df, syn_split_df, label_columns, split_types=["train", "valid"], config=None):
    """
    Analyzes the label counts for real and synthetic datasets across specified split types and 
    outputs the results as a dictionary.
    """
    results = {}

    for split_type in split_types:
        # Analyze each split type
        analysis_results = []

        for col in label_columns:
            real_data = real_split_df[real_split_df.seed0 == split_type]
            syn_data = syn_split_df[syn_split_df.seed0 == split_type]

            real_counts = real_data[col].value_counts()
            syn_counts = syn_data[col].value_counts()

            # Convert counts into row-wise format
            for label in set(real_counts.index).union(set(syn_counts.index)):
                analysis_results.append({
                    'label_column': col,
                    'label_value': label,
                    'real_count': real_counts.get(label, 0),
                    'syn_count': syn_counts.get(label, 0)
                })

        # Convert results into a DataFrame for processing
        results_df = pd.DataFrame(analysis_results)
        label_summary = results_df.groupby("label_column")[["real_count", "syn_count"]].sum().reset_index()

        # Add results to dictionary
        for _, row in label_summary.iterrows():
            label_col = row['label_column']
            if label_col not in results:
                results[label_col] = {}

            results[label_col][split_type] = {
                'real_count': row['real_count'],
                'syn_count': row['syn_count'],
                'pred_num': min(row['real_count'], row['syn_count'])
            }

    return results

def reduce_sample_rows_for_task(split_df, seed_column, task_column, pred_num_samples, config):
    """
    Reduces the number of samples based on the prediction number and updates the seed column.
    """
    np.random.seed(0)
    null_task_rows = split_df[(split_df[seed_column].isin(["train", "valid"])) & (split_df[task_column].isna())]
    split_df.loc[null_task_rows.index, seed_column] += "_discarded"
    return split_df[(split_df[seed_column].isin(["train", "valid", "test"])) & (split_df[task_column].notna())]

def process_task_and_save(task, task_df, base_datetime, prediction_time, label_root, config):
    """
    Processes and saves task data for specific tasks.
    """
    pid_col = config['pid_column']
    adjusted_datetime = base_datetime + pd.Timedelta(hours=prediction_time)
    task_df['prediction_time'] = adjusted_datetime
    task_df = task_df.rename(columns={pid_col: "subject_id", f'label_{task}': 'boolean_value'})
    task_df = task_df[["subject_id", "prediction_time", "boolean_value"]].dropna(subset=['boolean_value'])
    task_df["prediction_time"] = pd.to_datetime(task_df["prediction_time"], errors="coerce")
    task_df["boolean_value"] = task_df["boolean_value"].astype(int)
    
    # Save the processed task data
    label_path = os.path.join(label_root, "tasks", task)
    os.makedirs(label_path, exist_ok=True)
    task_df = task_df.sort_values(by=["subject_id", "prediction_time"])
    task_df.to_parquet(os.path.join(label_path, "0.parquet"))
    print(f"Data saved to {label_path}/0.parquet")
    return task_df

def main(config):
    pid_col = config['pid_column']
    time_col = config['time_column']

    # config based mapping
    lab_test_bins = config['lab_test_bins']
    labels = config['lab_labels']
    task2string = config['task_string']
    task_config = config['task_config']

    obs_size = config['obs_size']
    ehr = config['ehr']
    real_data_root = config['real_data_root']
    syn_data_root = config['syn_data_root']
    seed_column = f"seed{config['seed']}"

    table_names = config['table_names']
    split_map = {"train": "train", "valid": "tuning", "test": "held_out"}
    task_per_table = {t: task2string.get(t, []) for t in table_names}

    # Convert EHR to MEDS
    real_meds, syn_meds, real_df, syn_df = convert_tables_to_meds(
        config,
        ehr=ehr,
        table_names=table_names,
        real_data_root=real_data_root,
        syn_data_root=syn_data_root,
        base_datetime_str="2072-01-01 00:00:00"
    )

    # Process real data
    original_real_split_df = pd.read_csv(os.path.join(real_data_root, config["split_file_name"]))
    original_real_split_df = original_real_split_df.reset_index().rename(columns={"index": pid_col})
    real_split_df = process_and_split_data(
        real_df, obs_size, task_config, ehr, task2string, lab_test_bins, labels, original_real_split_df[[pid_col, 'pid', seed_column]]
    )
    save_split_data(real_data_root, real_split_df, seed_column, real_meds, split_map, is_real=True, config=config)

    # Process synthetic data
    syn_split_df = process_and_split_data(
        syn_df, obs_size, task_config, ehr, task2string, lab_test_bins, labels
    )
    save_split_data(syn_data_root, syn_split_df, seed_column, syn_meds, split_map, is_real=False, config=config)

    # Process test split for real data with remapped IDs
    real_test_stay_ids, new_stay_ids = process_test_split(syn_data_root, real_split_df, seed_column, real_meds, syn_split_df[pid_col].max() + 1, config)

    label_columns = [col for col in real_split_df.columns if 'label' in col]
    results_dict = analyze_label_counts(real_split_df, syn_split_df, label_columns, config=config)

    # Process tasks per table
    for table_name, tasks in task_per_table.items():
        print(f"\n--- Processing table: {table_name} ---")
        for task in tasks:
            print(f"\nProcessing task: {task}")
            task_column = f"label_{task}"

            temp_real_split_df = reduce_sample_rows_for_task(real_split_df.copy(), seed_column, task_column, results_dict, config)
            temp_syn_split_df = reduce_sample_rows_for_task(syn_split_df.copy(), seed_column, task_column, results_dict, config)

            # temp_real_split_df = real_split_df.copy()
            # temp_syn_split_df = syn_split_df.copy()

            temp_real_test = real_split_df[real_split_df[pid_col].isin(real_test_stay_ids)][["stay_id", task_column]]
            temp_real_test = temp_real_test[temp_real_test[task_column].notna()]

            # Process real and synthetic task data
            real_label_df = pd.concat([temp_real_split_df[["stay_id", task_column]], temp_real_test], axis=0)
            real_task_df = process_task_and_save(task, real_label_df, base_datetime, prediction_time, real_data_root, config)
            
            temp_real_test["stay_id"] = temp_real_test["stay_id"].map(new_stay_ids)
            syn_label_df = pd.concat([temp_syn_split_df[["stay_id", task_column]], temp_real_test], axis=0)
            syn_task_df = process_task_and_save(task, syn_label_df, base_datetime, prediction_time, syn_data_root, config)

if __name__ == "__main__":
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
        seed=args.seed,
    )
    main(config)
