import numpy as np
import pandas as pd
import os


def transform_ehr_to_meds(config, table_name, data_frame):
    pid_col = config['pid_column']
    time_col = config['time_column']
    id_column = config['item_column'][table_name]

    exclude_columns = [pid_col, time_col, id_column]
    code_prefix = table_name.upper()

    # Special handling for patientweight
    special_weight_df = pd.DataFrame()
    if 'patientweight' in data_frame.columns:
        special_weight_df = data_frame[[pid_col, time_col, 'patientweight']].copy()
        special_weight_df['code'] = "SUBJECT_WEIGHT_AT_INFUSION//KG"
        special_weight_df.rename(columns={'patientweight': 'numeric_value'}, inplace=True)
        data_frame = data_frame.drop(columns=['patientweight'])
    else:
        # Identify numeric and categorical columns
        numeric_cols = data_frame.select_dtypes(include=[np.number]).columns.difference(exclude_columns)
        categorical_cols = data_frame.select_dtypes(exclude=[np.number]).columns.difference(exclude_columns)
        
        # Melt numeric columns
        if not numeric_cols.empty:
            numeric_melted = data_frame.melt(
                id_vars=[pid_col, time_col, id_column],
                value_vars=numeric_cols,
                var_name='variable',
                value_name='numeric_value'
            )
        else:
            numeric_melted = pd.DataFrame(columns=[pid_col, time_col, id_column, 'variable', 'numeric_value'])
        
        # Melt categorical columns and modify code directly
        if not categorical_cols.empty:
            categorical_melted = data_frame.melt(
                id_vars=[pid_col, time_col, id_column],
                value_vars=categorical_cols,
                var_name='variable',
                value_name='value'
            )
            # Save in the form of column name + value in the 'code' column of categorical_melted
            categorical_melted['code'] = (
                code_prefix + "//" +
                categorical_melted[id_column].astype(str) + "//" +
                categorical_melted['variable'] + "//" + categorical_melted['value'].astype(str)
            )
            categorical_melted = categorical_melted[[pid_col, time_col, 'code']]
        else:
            categorical_melted = pd.DataFrame(columns=[pid_col, time_col, 'code'])
        
        # Construct 'code' for numeric columns
        numeric_melted['code'] = (
            code_prefix + "//" +
            numeric_melted[id_column].astype(str) + "//" +
            numeric_melted['variable']
        )
        
        # Select required columns for numeric_melted
        numeric_melted = numeric_melted[[pid_col, time_col, 'code', 'numeric_value']]
        
        # Combine numeric and categorical data
        final_df = pd.concat([numeric_melted, categorical_melted, special_weight_df], ignore_index=True)
    return final_df
    
def convert_tables_to_meds(config, table_names, real_data_root, syn_data_root, base_datetime_str="2072-01-01 00:00:00"):
    base_datetime = pd.to_datetime(base_datetime_str)
    real_meds_list = []
    syn_meds_list = []
    real_df = {}
    syn_df = {}
    time_col = config['time_column']
    pid_col = config['pid_column']

    for table_name in table_names:
        real_data = pd.read_csv(os.path.join(real_data_root, f"{table_name}.csv"))
        syn_data = pd.read_csv(os.path.join(syn_data_root, f"{table_name}.csv"))
        
        # Exception handling for timestamp, value, etc.
        if 'timestamp' in real_data.columns:
            real_data.rename(columns={'timestamp': time_col}, inplace=True)
            syn_data.rename(columns={'timestamp': time_col}, inplace=True)

        if 'value' in real_data.columns: # The column 'value' already exists.
            real_data.rename(columns={'value': 'lab_value'}, inplace=True)
            syn_data.rename(columns={'value': 'lab_value'}, inplace=True)

        real_df[table_name] = real_data.copy()
        syn_df[table_name] = syn_data.copy()

        real_data[time_col] = real_data[time_col] // 10 * 10
        real_data['converted_time'] = base_datetime + pd.to_timedelta(real_data[time_col], unit='m')
        real_data[time_col] = real_data['converted_time']
        real_data = real_data.drop(columns=['converted_time'])

        syn_data['converted_time'] = base_datetime + pd.to_timedelta(syn_data[time_col], unit='m')
        syn_data[time_col] = syn_data['converted_time']
        syn_data = syn_data.drop(columns=['converted_time'])

        real_sub_meds = transform_ehr_to_meds(table_name, real_data, config)
        syn_sub_meds = transform_ehr_to_meds(table_name, syn_data, config)

        real_meds_list.append(real_sub_meds)
        syn_meds_list.append(syn_sub_meds)

    real_meds = pd.concat(real_meds_list, axis=0).rename(columns={pid_col: "subject_id"})
    syn_meds = pd.concat(syn_meds_list, axis=0).rename(columns={pid_col: "subject_id"})

    real_meds = real_meds.sort_values(by=["subject_id", time_col])
    syn_meds = syn_meds.sort_values(by=["subject_id", time_col])
    print("converting tables to meds is done")


    return real_meds, syn_meds, real_df, syn_df

def process_data(config, real_meds, syn_meds, real_data_root, min_code_inclusion_count=1000):
    
    # Load and filter splits
    splits_file = os.path.join(real_data_root, config["split_file_name"])
    splits = pd.read_csv(splits_file).reset_index()[["index", config['split_column']]]
    splits = splits[splits["index"].isin(real_meds.subject_id.unique())]

    # Extract training stay IDs
    train_stay_ids = splits[splits.seed0 == "train"]["index"].values

    # Match real and synthetic training stay IDs
    real_train_stay_ids = real_meds[real_meds.subject_id.isin(train_stay_ids)].subject_id.unique()
    real_meds = real_meds[real_meds.subject_id.isin(real_train_stay_ids)]

    # Step 1: Group by 'code' and count unique 'subject_id'
    code_stay_id_counts = real_meds.groupby('code')['subject_id'].nunique()

    # Step 2: Identify codes with sufficient occurrences
    valid_codes = code_stay_id_counts[code_stay_id_counts >= min_code_inclusion_count].index

    # Step 3: Filter `real_meds` and `syn_meds` for valid codes
    real_meds = real_meds[real_meds['code'].isin(valid_codes)]
    syn_meds = syn_meds[syn_meds['code'].isin(valid_codes)]

    real_meds = real_meds.sort_values(by=["subject_id", config['time_column']])
    syn_meds = syn_meds.sort_values(by=["subject_id", config['time_column']])
    
    # Print number of unique codes in each dataset
    print("real_meds.code.nunique(): ", real_meds.code.nunique())
    print("syn_meds.code.nunique(): ", syn_meds.code.nunique())

    return real_meds, syn_meds

