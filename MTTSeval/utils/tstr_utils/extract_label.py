import numpy as np
import pandas as pd

# All config is now expected to come from the new get_config loader in utils.configs.
# This means config is always unified and dataset-agnostic.

# In all functions, replace 'stay_id' and 'time' with config['pid_column'] and config['time_column']

def get_last_non_null_values(df, config):
    """
    Extract the last non-null values for specific tasks within a specified observation size.
    """
    lab_config = config["task_config"]["lab"]
    TASK2STRING = config["task_string"]
    itemid_col = lab_config["itemid_col"]
    value_col = lab_config["value_col"]
    obs_size = config["obs_size"]

    # Create a dictionary to map each item to its corresponding task
    item_to_task = {}
    for task, itemids in TASK2STRING.items():
        for itemid in itemids:
            item_to_task[itemid] = task
    
    # Filter by time and relevant itemids
    relevant_itemids = set(item_to_task.keys())
    filtered_df = df[(df[config['time_column']] > (obs_size // 2) * 60) & (df[itemid_col].isin(relevant_itemids))].copy()

    # Map the item IDs in the dataframe to the corresponding task
    filtered_df['task'] = filtered_df[itemid_col].map(item_to_task)

    # Sorting to ensure last value is the latest
    filtered_df = filtered_df.sort_values(by=config['time_column'])

    # Aggregating last non-null values for each task per stay_id
    last_values = filtered_df.groupby([config['pid_column'], 'task']).agg({
        value_col: lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None
    }).reset_index()

    return last_values


def apply_binning(last_values, config):
    """
    Apply binning to the last values and generate labels for each task.
    """
    LAB_TEST_BINS = config["lab_test_bins"]
    LABELS = config["lab_labels"]
    value_col = config["task_config"]["lab"]["value_col"]
    
    # Prepare an empty DataFrame to collect results
    results = pd.DataFrame({'stay_id': last_values['stay_id'].unique()})
    
    # Apply binning for each test
    for test, bins in LAB_TEST_BINS.items():
        # Filter relevant values by task
        relevant_values = last_values[last_values['task'] == test]
        
        def get_label(row):
            if not pd.isnull(row[value_col]):
                return pd.cut([row[value_col]], bins=bins, labels=LABELS[test])[0]
            return None
        
        # Apply binning and create a new column with labels
        relevant_values = relevant_values.copy()
        relevant_values['label_' + test] = relevant_values.apply(get_label, axis=1)
        
        # Merge the results with the main DataFrame
        results = results.merge(relevant_values[['stay_id', 'label_' + test]], on='stay_id', how='left')
    
    return results


def extract_labels(config, df, target_col, obs_size, target_tasks):
    """
    Generate binary labels for each task within the specified observation window.
    """

    def process_itemid_data(df, itemid, obs_size):
        itemid = itemid.lower()
        filtered_df = df[df[target_col].str.contains(itemid, case=False, na=False)]
        
        # Define time cutoffs
        time_cutoff = (obs_size // 2) * 60
        next_window_cutoff = 2 * time_cutoff

        # Identify stay_ids within the following 6 hours
        stay_ids_after_6_hours = set(filtered_df[
            (filtered_df[config['time_column']] > time_cutoff) & (filtered_df[config['time_column']] <= next_window_cutoff)
        ][config['pid_column']].unique())

        return stay_ids_after_6_hours

    # Prepare label data
    label_data = {'stay_id': list(df[config['pid_column']].unique())}
    
    for itemid in target_tasks:
        # Process each itemid
        stay_ids_after_6_hours = process_itemid_data(df, itemid, obs_size)
        label_data[f'label_{itemid}'] = [
            1 if stay_id in stay_ids_after_6_hours else 0 for stay_id in label_data['stay_id']
        ]
    
    # Convert to DataFrame
    labels_df = pd.DataFrame(label_data)
    return labels_df


def process_dataframes(dataframes, config, stay_id_df=None):
    """
    Processes multiple DataFrames and combines them into a single output DataFrame.
    """
    lab_config = config["task_config"]["lab"]
    obs_size = config["obs_size"]
    
    # Process lab data
    last_lab_values = get_last_non_null_values(
        dataframes[lab_config["table_name"]], config
    )
    lab_label_df = apply_binning(last_lab_values, config)
    print("Lab data processed.")

    # Process input data
    input_label_df = extract_labels(
        config,
        dataframes[config["task_config"]['input']['table_name']],
        config["task_config"]['input']['itemid_col'],
        obs_size,
        target_tasks=config["task_config"]['input']['task']
    )
    print("Input data processed.")

    # Process medication data
    med_label_df = extract_labels(
        config,
        dataframes[config["task_config"]['med']['table_name']],
        config["task_config"]['med']['itemid_col'],
        obs_size,
        target_tasks=config["task_config"]['med']['task']
    )
    print("Medication data processed.")

    # Compute row counts for stay_id from filtered DataFrame
    threshold_time = (obs_size // 2) * 60
    filtered = pd.concat([
        df[df[config['time_column']] <= threshold_time][[config['pid_column']]]
        for df in dataframes.values()
    ])
    counts = filtered[config['pid_column']].value_counts().reset_index()
    counts.columns = [config['pid_column'], 'total_half_event']

    # Either create a new output_df or update the existing one
    if stay_id_df is None:
        # Collect all unique stay_id from dataframes
        total_stay_ids = set()
        for df in dataframes.values():
            total_stay_ids.update(df[config['pid_column']].unique())
        total_stay_id_df = pd.DataFrame({config['pid_column']: list(total_stay_ids)})

        # Merge counts with total_stay_id_df and handle missing values
        total_stay_id_df = total_stay_id_df.merge(counts, on=config['pid_column'], how='left').fillna(0)
        total_stay_id_df['total_half_event'] = total_stay_id_df['total_half_event'].astype(int)
        
        split_df = total_stay_id_df \
            .merge(input_label_df, on=config['pid_column'], how='left') \
            .merge(med_label_df, on=config['pid_column'], how='left') \
            .merge(lab_label_df, on=config['pid_column'], how='left')
    else:
        # Merge counts with total_stay_id_df and handle missing values
        total_stay_id_df = stay_id_df.merge(counts, on=config['pid_column'], how='left').fillna(0)
        total_stay_id_df['total_half_event'] = total_stay_id_df['total_half_event'].astype(int)
        
        split_df = total_stay_id_df \
            .merge(input_label_df, on=config['pid_column'], how='left') \
            .merge(med_label_df, on=config['pid_column'], how='left') \
            .merge(lab_label_df, on=config['pid_column'], how='left')

    # Identify label columns and set values to None where row_count <= 5
    label_columns = [col for col in split_df.columns if 'label' in col]
    print(f"Total nullified label values: {len(split_df[split_df['total_half_event'] <= 5])}")
    split_df.loc[split_df['total_half_event'] <= 5, label_columns] = None
    
    # Return the processed DataFrame
    if stay_id_df is None:
        split_col = config['split_column']
        split_df = split_train_valid(split_df.copy(), split_col=split_col)

    return split_df


def split_train_valid(df, seed=0, test_size=0.11, split_col=None):
    np.random.seed(seed)  # Set random seed for reproducibility
    # Shuffle the indices of the DataFrame
    df = df.copy()
    shuffled_indices = np.random.permutation(df.index)
    
    # Calculate the split point
    split_point = int(len(shuffled_indices) * (1 - test_size))
    
    # Split the indices into train and valid
    train_indices = shuffled_indices[:split_point]
    valid_indices = shuffled_indices[split_point:]
    
    # Assign "train" and "valid" to the seed column
    df.loc[train_indices, split_col] = "train"
    df.loc[valid_indices, split_col] = "valid"
    
    return df