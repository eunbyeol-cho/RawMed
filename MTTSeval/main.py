import os
import json
import argparse
import pandas as pd
import numpy as np
from evaluation.eval_corr import CorrelationAnalyzer
from evaluation.eval_stat import StatisticalDistributionAnalyzer
from evaluation.eval_time import TimeAnalyzer
from evaluation.eval_utility import run_tstr
from evaluation.eval_predsim import run_pred_similarity
from evaluation.eval_cond import run_cond
from evaluation.eval_next_event_predict import run_next_event_predict
from utils.convert_table_to_text import load_tables
from utils.configs import get_config


def analyze_statistics(stat_analyzer, output_dir):
    """
    Perform statistical analysis and save the results.
    """
    stat_summary = stat_analyzer.generate_statistics_report()
    stat_summary.to_csv(os.path.join(output_dir, "stat_summary.csv"), index=False)


# Check if NaN, inf, None exists
def check_invalid_values(item_specific_results):
    has_nan = False
    has_inf = False
    has_none = False

    for key, result in item_specific_results.items():
        if "mu_abs" in result:
            value = result["mu_abs"]

            if value is None:
                has_none = True
                print(f"[WARNING] Detected None value: {key}")

            elif isinstance(value, float) and np.isnan(value):
                has_nan = True
                print(f"[WARNING] Detected NaN value: {key}")

            elif isinstance(value, float) and np.isinf(value):
                has_inf = True
                print(f"[WARNING] Detected Inf value: {key}")

    return has_nan, has_inf, has_none

def analyze_correlation(corr_analyzer, real_df, syn_df, output_dir):
    """
    Perform correlation analysis and save the results.
    """
    # General Correlation Analysis
    general_results = corr_analyzer.analyze(real_df, syn_df, plot=True)
    general_results["real_correlation_matrix"].to_csv(
        os.path.join(output_dir, "real_corr_matrix.csv"), index=True
    )
    general_results["synthetic_correlation_matrix"].to_csv(
        os.path.join(output_dir, "synthetic_corr_matrix.csv"), index=True
    )

    # Item-Specific Correlation Analysis
    item_specific_results = corr_analyzer.analyze_by_item(real_df, syn_df, plot=False)
    pd.DataFrame.from_dict(item_specific_results, orient="index").to_csv(
        os.path.join(output_dir, "item_corr.csv")
    )
    # has_nan, has_inf, has_none = check_invalid_values(item_specific_results)

    # Save overall metrics
    overall_metrics = {
        "mean_mu_abs": round(np.mean([r["mu_abs"] for r in item_specific_results.values()]), 3),
        "mean_cor_acc": round(np.mean([r["cor_acc"] for r in item_specific_results.values()]), 3),
        "general_mu_abs": general_results["mu_abs"],
        "general_cor_acc": general_results["cor_acc"]
    }
    with open(os.path.join(output_dir, "overall_metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=2)

def analyze_time(real_times, syn_times, config, output_dir):
    """
    Perform time analysis and save the results.
    """
    pid_col = config["pid_column"]
    time_col = config["time_column"]
    real_combined = pd.concat(real_times)[[pid_col, time_col]].sort_values(by=[pid_col, time_col]).reset_index(drop=True)
    syn_combined = pd.concat(syn_times)[[pid_col, time_col]].sort_values(by=[pid_col, time_col]).reset_index(drop=True)
    
    grouped_real_times = real_combined.groupby(pid_col)[time_col].apply(list).tolist()
    grouped_syn_times = syn_combined.groupby(pid_col)[time_col].apply(list).tolist()

    time_analyzer = TimeAnalyzer(config["ehr"], grouped_real_times, grouped_syn_times)
    time_summary = time_analyzer.analyze(plot_type=None)
    time_summary.to_csv(os.path.join(output_dir, "time_summary.csv"), index=True)


def eval_stat_corr_time(config):
    # Load tables and preprocess
    ehr = config["ehr"]
    table_names = config["table_names"]
    real_data_root = config["real_data_root"]
    syn_data_root = config["syn_data_root"]
    split_col = config["split_column"]

    output_data_root = config["output_data_root"]
    os.makedirs(output_data_root, exist_ok=True)

    # Load predefined data
    real_dfs, syn_dfs = load_tables(config)
    col_type = pd.read_pickle(os.path.join(real_data_root, config["col_type"]))
    splits = pd.read_csv(os.path.join(real_data_root, config["split_file_name"])).reset_index()
    train_indices = splits[splits[split_col] == "train"]['index']

    # Configuration
    real_times = []
    syn_times = []

    for table_name in table_names:  
        output_dir = os.path.join(output_data_root, table_name)
        os.makedirs(output_dir, exist_ok=True)

        real_df = real_dfs[table_name]
        real_df = real_df[real_df[config["pid_column"]].isin(train_indices)]
        syn_df = syn_dfs[table_name]
        
        real_times.append(real_df)
        syn_times.append(syn_df)

        print(f"{table_name} Shapes - Real: {real_df.shape}, Synthetic: {syn_df.shape}")

        # Statistical Analysis
        stat_analyzer = StatisticalDistributionAnalyzer(
            config, real_df, syn_df, table_name,
            col_type[table_name]["numeric_columns"],
            col_type[table_name]["categorical_columns"]
        )
        analyze_statistics(stat_analyzer, output_dir)

        # Correlation Analysis
        corr_analyzer = CorrelationAnalyzer(
            config, table_name,
            col_type[table_name]["numeric_columns"],
            col_type[table_name]["categorical_columns"]
        )
        analyze_correlation(corr_analyzer, real_df, syn_df, output_dir)

    # Time Analysis
    time_output_dir = os.path.join(output_data_root, "time")
    os.makedirs(time_output_dir, exist_ok=True)
    analyze_time(real_times, syn_times, config, time_output_dir)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ehr', type=str, required=True, choices=['mimiciv', 'eicu'], help='EHR dataset')
    parser.add_argument('--obs_size', type=int, default=12)
    parser.add_argument('--real_data_root', type=str, required=True)
    parser.add_argument('--syn_data_root', type=str, required=True)
    parser.add_argument('--output_data_root', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Load config using new loader
    config = get_config(
        ehr=args.ehr,
        obs_size=args.obs_size,
        real_data_root=args.real_data_root,
        syn_data_root=args.syn_data_root,
        output_data_root=args.output_data_root,
        seed=args.seed,
    )
    
    if args.seed == 0:
        # eval_stat_corr_time(config)
        # run_pred_similarity(config)
        # run_next_event_predict(config)
        run_tstr(config)
        # run_cond(config)
    else:
        # eval_stat_corr_time(config) # doesn't change with seed
        # run_pred_similarity(config)
        # run_next_event_predict(config)
        run_tstr(config)
        # run_cond(config)