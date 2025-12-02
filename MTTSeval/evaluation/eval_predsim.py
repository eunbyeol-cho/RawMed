import os
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import random
import xgboost as xgb
from utils.configs import get_config
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    smape_value = 2.0 * np.mean(np.abs(actual - predicted) / denominator) * 100
    return smape_value

def clean_target_data(target_data):
    mask = target_data.notna()
    cleaned_target_data = target_data[mask]
    return cleaned_target_data, mask

def determine_task_type(target_data, is_numerical):
    if is_numerical:
        return 'regression'
    else:
        if target_data.nunique() == 2:
            return 'binary_classification'
        elif target_data.dtype == 'object' or target_data.nunique() > 2:
            return 'multiclass_classification'        
        else:
            raise ValueError("Unsupported data type for target")

def preprocess_features(X):
    """Handle preprocessing of both categorical and numeric features."""
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('scaler', StandardScaler())]), numeric_cols),
            
            ('cat', Pipeline(steps=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
        ]
    )
    return preprocessor

def remove_unseen_classes(X_train, y_train, X_test, y_test):
    """Remove test instances with unseen classes."""
    valid_classes = np.unique(y_train)
    mask = np.isin(y_test, valid_classes)
    return X_test[mask], y_test[mask]

def get_model(task_type, use_gpu=False, num_classes=None, seed=0):
    """Select model based on task type and model type with defined XGBoost parameters."""
    # Define XGBoost parameters
    params = {
        'objective': 'binary:logistic' if task_type == 'binary_classification' else 'multi:softmax' if task_type == 'multiclass_classification' else 'reg:squarederror',
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'predictor': 'gpu_predictor' if use_gpu else 'cpu_predictor',
        'eval_metric': 'logloss' if task_type == 'binary_classification' else 'mlogloss' if task_type == 'multiclass_classification' else 'rmse',
        'max_bin': 256,
        'nthread': -1,
        'seed': seed,
        'subsample': 0.9,
    }

    # Set num_class for multiclass classification
    if task_type == 'multiclass_classification' and num_classes:
        params['num_class'] = num_classes

    if task_type in ['binary_classification', 'multiclass_classification']:
        return xgb.XGBClassifier(**params, n_estimators=100, use_label_encoder=False, random_state=seed)
    elif task_type == 'regression':
        return xgb.XGBRegressor(**params, n_estimators=100, random_state=seed)


def remove_rare_classes(X, y, threshold):
    """Remove classes with fewer than `threshold` percentage occurrences."""
    total_samples = len(y)
    min_samples = int(total_samples * threshold)  # Calculate the minimum sample count based on the threshold

    class_counts = y.value_counts()  # Use value_counts to handle non-integer categorical labels
    valid_classes = class_counts[class_counts >= min_samples].index  # Get valid classes with enough samples
    mask = y.isin(valid_classes)  # Create a mask for valid classes
    
    # Calculate statistics
    total_classes = len(y.unique())
    removed_classes = total_classes - len(valid_classes)
    
    removed_samples = total_samples - np.sum(mask)
    
    # Print detailed statistics
    print(f"Removed {removed_classes} out of {total_classes} classes (less than {min_samples} samples).")
    print(f"Removed {removed_samples} out of {total_samples} samples due to rare classes.")
    
    # Return filtered DataFrame instead of numpy arrays
    return X[mask], y[mask]


def run_task(target_col, train_data, real_test, mode, real_classes=None, seed=0, numeric_columns=None, categorical_columns=None):
    # Input features: exclude the target column
    features_train = train_data.drop(columns=[target_col])
    features_real_test = real_test.drop(columns=[target_col])

    # Extract target data
    target_data_train = train_data[target_col]
    target_data_real_test = real_test[target_col]

    # Clean target data (remove NaNs and infinities)
    target_data_train, mask_train = clean_target_data(target_data_train)
    features_train = features_train[mask_train]  # Remove corresponding feature rows for training data
    print(f"Training data size before rare class removal: {len(target_data_train)}")
    
    # Check if target_data has only one unique value
    if target_data_train.nunique() == 1:
        print(f"Skipping target column {target_col} as it has only one unique value in training data.")
        return None, None
    
    target_data_real_test, mask_real_test = clean_target_data(target_data_real_test)
    features_real_test = features_real_test[mask_real_test]  # Remove corresponding feature rows for real test data

    # Determine task type
    is_numerical = target_col in numeric_columns
    task_type = determine_task_type(target_data_train, is_numerical)
    print(f"Running {task_type} task for target column: {target_col}")
    
    if task_type in ['binary_classification', 'multiclass_classification']:
        if mode == "tstr":
            print(f"Filtering data to retain only valid classes: {len(real_classes)}")
            valid_train_mask = target_data_train.isin(real_classes)
            target_data_train = target_data_train[valid_train_mask]
            features_train = features_train[valid_train_mask]        
        else:
            # Apply the remove_rare_classes function
            features_train, target_data_train = remove_rare_classes(features_train, target_data_train, threshold=0.001)
    
        print(f"Training data size after rare class removal: {len(target_data_train)}")
        
        # Filter real test data based on valid classes in training data
        valid_test_mask = target_data_real_test.isin(target_data_train)
        target_data_real_test = target_data_real_test[valid_test_mask]
        features_real_test = features_real_test[valid_test_mask]
        print(f"Test data size after filtering based on training data: {len(valid_test_mask)} -> {len(target_data_real_test)}")

    # Skip if no valid test samples remain
    if target_data_real_test.empty:
        print(f"Skipping target column {target_col} as no valid test labels exist in the training set.")
        return None, None
    
    # Fit LabelEncoder on the combined dataset
    if task_type in ['binary_classification', 'multiclass_classification']:
        combined_target_data = pd.concat([target_data_train, target_data_real_test])
        le = LabelEncoder()
        le.fit(combined_target_data)
        target_data_train = le.transform(target_data_train)
        target_data_real_test = le.transform(target_data_real_test)
        valid_classes = le.classes_
    else:
        valid_classes = None
    
    # Preprocessing pipeline
    preprocessor = preprocess_features(features_train)  # Assuming both datasets have the same preprocessing needs
    X_train = preprocessor.fit_transform(pd.DataFrame(features_train, columns=features_train.columns))
    X_test = preprocessor.transform(pd.DataFrame(features_real_test, columns=features_train.columns))
    
    num_classes = len(np.unique(target_data_train))
    model = get_model(task_type, use_gpu=True, num_classes=num_classes, seed=seed)


    if task_type in ['binary_classification', 'multiclass_classification']:
        print(f"Task type: {task_type} using XGBoost with GPU in {mode} mode")
        model.fit(X_train, target_data_train)
        y_pred = model.predict(X_test)
        report = classification_report(target_data_real_test, y_pred, output_dict=True)
        results = {
            'column': target_col,
            'task_type': task_type,
            'mode': mode,
            'accuracy': report['accuracy'],
            'macro avg f1-score': report['macro avg']['f1-score'],
            'weighted avg f1-score': report['weighted avg']['f1-score'],
            'support': report['macro avg']['support']
        }
        
    elif task_type == 'regression':
        print(f"Task type: Regression using XGBoost with GPU in {mode} mode")
        model.fit(X_train, target_data_train)
        y_pred = model.predict(X_test)
        mse = mean_absolute_error(target_data_real_test, y_pred)
        smape_val = smape(target_data_real_test, y_pred)
        results = {
            'column': target_col,
            'task_type': task_type,
            'mode': mode,
            'mse': mse,
            'smape': smape_val
        }
    return results, valid_classes

def run_pred_similarity(config):
    pid_col = config['pid_column']
    time_col = config['time_column']
    split_col = config['split_column']
    ehr = config['ehr']
    obs_size = config['obs_size']
    real_data_root = config['real_data_root']
    syn_data_root = config['syn_data_root']
    seed = config['seed']
    table_names = config['table_names']
    output_data_root = config['output_data_root']

    set_seed(seed)
    col_dtype = pd.read_pickle(os.path.join(real_data_root, f"{ehr}_col_dtype.pickle"))

    for table_name in table_names:
        real_data = pd.read_csv(os.path.join(real_data_root, f"{table_name}.csv"))
        syn_train = pd.read_csv(os.path.join(syn_data_root, f"{table_name}.csv")).reset_index(drop=True)
        
        numeric_columns = col_dtype[table_name]["numeric_columns"]
        categorical_columns = col_dtype[table_name]["categorical_columns"]

        original_real_split_df = pd.read_csv(os.path.join(real_data_root, config["split_file_name"]))
        original_real_split_df = original_real_split_df.reset_index().rename(columns={"index": pid_col})
        test_stay_ids = original_real_split_df[original_real_split_df[split_col] == "test"][pid_col].unique()
        train_stay_ids = original_real_split_df[original_real_split_df[split_col] == "train"][pid_col].unique()

        real_train = real_data[real_data[pid_col].isin(train_stay_ids)]
        real_test = real_data[real_data[pid_col].isin(test_stay_ids)]
        
        # Remove duplicates, excluding pid_col and time_col
        cols_for_dedup = [col for col in real_data.columns if col not in [pid_col, time_col]]
        real_train = real_train.drop_duplicates(subset=cols_for_dedup)
        real_test = real_test.drop_duplicates(subset=cols_for_dedup)
        syn_train = syn_train.drop_duplicates(subset=cols_for_dedup)

        train_combined = pd.concat([real_train[cols_for_dedup], syn_train[cols_for_dedup]]).drop_duplicates()

        merged = pd.merge(
            real_test,                
            train_combined,
            on=cols_for_dedup,        
            how='left',
            indicator=True
        )

        real_test_clean = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        print("Before removing duplicates, real_test size:", len(real_test))
        print("After removing duplicates, real_test size:", len(real_test_clean))

        real_test = real_test_clean.copy()
        print(real_train.shape, real_test.shape, syn_train.shape)
        
        # Remove columns if needed
        remove_columns = True
        if remove_columns:
            columns_to_drop = [pid_col, time_col]
            syn_train = syn_train.drop(columns=columns_to_drop)
            real_train = real_train.drop(columns=columns_to_drop)
            real_test = real_test.drop(columns=columns_to_drop)
            target_cols = [col for col in real_train.columns]
        else:
            target_cols = [col for col in real_train.columns if col not in [pid_col, time_col]]

        final_results = []
        for target_col in target_cols:
            print(target_col)
            train_data = real_train.copy()
            results, real_classes= run_task(target_col, train_data, real_test, mode="trtr", real_classes=None, seed=seed, numeric_columns=numeric_columns, categorical_columns=categorical_columns)
            if results:
                final_results.append(results)
                print(results, '\n')

                train_data = syn_train.copy()
                results, _ = run_task(target_col, train_data, real_test, mode="tstr", real_classes=real_classes, seed=seed, numeric_columns=numeric_columns, categorical_columns=categorical_columns)
                final_results.append(results)
                print(results, '\n')
        
        final_results = [result for result in final_results if result is not None]
        pd.DataFrame(final_results).to_csv(os.path.join(output_data_root, f"{ehr}_{table_name}_{seed}.csv"), index=False)
    
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
    run_pred_similarity(config)
        
    
