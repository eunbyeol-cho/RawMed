import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings(action='ignore')

class StatisticalDistributionAnalyzer:
    def __init__(self, config, real_df, synthetic_df, table_name, numeric_columns, categorical_columns):
        self.real_df = real_df
        self.synthetic_df = synthetic_df
        self.table_name = table_name
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.item_column = config["item_column"][table_name]
        
    def calculate_ks_statistics(self, real_subset, synthetic_subset, column, verbose):
        """
        Calculate KS statistics for a numeric column.
        """
        # Calculate null percentages
        real_null_pct = real_subset[column].isnull().mean() * 100
        synthetic_null_pct = synthetic_subset[column].isnull().mean() * 100

        # Drop null values
        real_values = real_subset[column].dropna()
        synthetic_values = synthetic_subset[column].dropna()

        # Check for data availability
        if real_values.empty or synthetic_values.empty:
            if verbose:
                print(f"No data available in '{column}' for KS test.")
            return None, None, None, None, None, None, None

        # Perform KS test
        ks_stat, _ = ks_2samp(real_values, synthetic_values)
        real_mean, real_std = real_values.mean(), real_values.std()
        synthetic_mean, synthetic_std = synthetic_values.mean(), synthetic_values.std()

        return (
            round(ks_stat, 3),
            real_mean,
            real_std,
            synthetic_mean,
            synthetic_std,
            round(real_null_pct, 3),
            round(synthetic_null_pct, 3),
        )

    def calculate_js_divergence(self, real_subset, synthetic_subset, column, verbose):
        """
        Calculate Jensen-Shannon divergence for a categorical column.
        """
        # Calculate null percentages
        real_null_pct = real_subset[column].isnull().mean() * 100
        synthetic_null_pct = synthetic_subset[column].isnull().mean() * 100

        # Drop null values
        real_values = real_subset[column].dropna()
        synthetic_values = synthetic_subset[column].dropna()

        # Check for data availability
        if real_values.empty or synthetic_values.empty:
            if verbose:
                print(f"No data available in '{column}' for JS divergence calculation.")
            return None, None, None

        # Calculate normalized value counts
        real_counts = real_values.value_counts(normalize=True).sort_index()
        synthetic_counts = synthetic_values.value_counts(normalize=True).sort_index()

        # Align indices
        real_counts, synthetic_counts = real_counts.align(synthetic_counts, fill_value=0)

        # Calculate JS divergence
        js_divergence = jensenshannon(real_counts, synthetic_counts)

        return (
            round(js_divergence, 3),
            round(real_null_pct, 3),
            round(synthetic_null_pct, 3),
        )
        
    def generate_statistics_report(self, num_features=None, top_k=50, freq_threshold=0.001, verbose=False, include_mean=True):
        """
        Generate statistics report for numeric and categorical columns.
    
        Args:
            num_features: Number of top conditions to analyze.
            top_k: Number of top items to include in statistics.
            freq_threshold: Minimum frequency threshold for items (as a fraction of the total dataset).
            verbose: Whether to print verbose output.
        """
        self.top_k = top_k
        self.freq_threshold_ratio = freq_threshold
        self.freq_threshold_count = len(self.real_df) * freq_threshold
        
        ks_results = []
        js_results = []
    
        # Determine top conditions
        if self.item_column:
                
            if num_features is None:
                num_features = self.real_df[self.item_column].nunique()
        
            conditions = ["TOTAL"] + list(
                self.real_df[self.item_column].value_counts().nlargest(num_features).index
            )
        
            self.frequent_item_count = len(
                self.real_df[self.item_column].value_counts()[lambda x: x >= self.freq_threshold_count].index
            )
        
            print(f"Number of items: {len(conditions) - 1}")
            print(f"Number of items with at least {freq_threshold*100:.1f}% frequency: {self.frequent_item_count}")

        else:
            conditions = ["TOTAL"] 

        for condition in tqdm(conditions, desc="Processing Conditions"):
            if condition == "TOTAL":
                real_subset = self.real_df.copy()
                synthetic_subset = self.synthetic_df.copy()
            else:
                real_subset = self.real_df[self.real_df[self.item_column] == condition]
                synthetic_subset = self.synthetic_df[
                    self.synthetic_df[self.item_column] == condition
                ]
    
            if verbose:
                print(
                    f"Condition: {condition} | Real Data Shape: {real_subset.shape} | Synthetic Data Shape: {synthetic_subset.shape}"
                )
    
            # KS test for numeric columns
            for col in self.numeric_columns:
                if (col == "patientweight") and (condition != "TOTAL"):
                    continue
                    
                ks_stat, real_mean, real_std, syn_mean, syn_std, real_null_pct, syn_null_pct = self.calculate_ks_statistics(
                    real_subset, synthetic_subset, col, verbose
                )
    
                if ks_stat is not None:
                    ks_results.append(
                        {
                            "Condition": condition,
                            "Column": col,
                            "Real Null %": real_null_pct,
                            "Syn Null %": syn_null_pct,
                            "Real Mean": real_mean,
                            "Syn Mean": syn_mean,
                            "Real Std": real_std,
                            "Syn Std": syn_std,
                            "KS Statistic": ks_stat,
                        }
                    )
    
            # JS divergence for categorical columns
            for col in self.categorical_columns:
                if (col == self.item_column) and (condition != "TOTAL"):
                    continue
                js_divergence, real_null_pct, syn_null_pct = self.calculate_js_divergence(
                    real_subset, synthetic_subset, col, verbose
                )
    
                if js_divergence is not None:
                    js_results.append(
                        {
                            "Condition": condition,
                            "Column": col,
                            "Real Null %": real_null_pct,
                            "Syn Null %": syn_null_pct,
                            "JS Divergence": js_divergence,
                        }
                    )
    
        # Pivot KS results
        if len(self.numeric_columns) != 0:
            ks_results_df = pd.DataFrame(ks_results)
            ks_pivot_df = ks_results_df.pivot(
                index="Condition",
                columns="Column",
                values=["Real Null %", "Syn Null %", "Real Mean", "Syn Mean", "KS Statistic"]
            )
            ks_pivot_df = ks_pivot_df.stack(level=0).unstack(level=1)
            ks_pivot_df = ks_pivot_df.reindex(conditions)
        else:
            ks_pivot_df = None
    
        # Pivot JS divergence results
        js_results_df = pd.DataFrame(js_results)
        js_pivot_df = js_results_df.pivot(
            index="Condition",
            columns="Column",
            values=["Real Null %", "Syn Null %", "JS Divergence"],
        )
        js_pivot_df = js_pivot_df.stack(level=0).unstack(level=1)
        js_pivot_df = js_pivot_df.reindex(conditions)
    
        summary_df = self.create_summary_table(ks_pivot_df, js_pivot_df, include_mean=include_mean)
        return summary_df
    
    def create_summary_table(self, ks_pivot_df, js_pivot_df, include_mean=True):
        """
        Create a summary table combining KS statistics and JS divergence results.
    
        Args:
            table_name: Name of the table being analyzed.
            ks_pivot_df: Pivot table for KS statistics.
            js_pivot_df: Pivot table for JS divergence.
        """
        summary_data = []
    
        def get_ks_statistics(col, df):
            """ Helper function to retrieve and round KS statistics """
            if col in df:
                return {
                    "ks_stats_total": round(df[col]["KS Statistic"].iloc[0], 3),
                    "ks_stats_topk": round(df[col]["KS Statistic"].iloc[1:1+self.top_k].mean(), 3),
                    "ks_stats_frequent": round(df[col]["KS Statistic"].iloc[1:1+self.frequent_item_count].mean(), 3),
                    "ks_stats_all": round(df[col]["KS Statistic"].iloc[1:].mean(), 3),
                    "real_null_total": round(df[col]["Real Null %"].iloc[0], 3),
                    "syn_null_total": round(df[col]["Syn Null %"].iloc[0], 3),
                }
            else:
                return None
    
        def get_js_statistics(col, df):
            """ Helper function to retrieve and round JS statistics """
            if col in df:
                return {
                    "js_div_total": round(df[col]["JS Divergence"].iloc[0], 3),
                    "js_div_topk": round(df[col]["JS Divergence"].iloc[1:1+self.top_k].mean(), 3),
                    "js_div_frequent": round(df[col]["JS Divergence"].iloc[1:1+self.frequent_item_count].mean(), 3),
                    "js_div_all": round(df[col]["JS Divergence"].iloc[1:].mean(), 3),
                    "real_null_total": round(df[col]["Real Null %"].iloc[0], 3),
                    "syn_null_total": round(df[col]["Syn Null %"].iloc[0], 3),
                }
            else:
                return None

        # Process categorical columns
        for col in self.categorical_columns:
            stats = get_js_statistics(col, js_pivot_df)
            summary_data.append({
                "Table": self.table_name,
                "Column": col,
                "Category": "Categorical",
                "Column-wise": stats["js_div_total"] if stats else None,
                f"Top {self.top_k} Items": stats["js_div_topk"] if stats else None,
                f"≥{self.freq_threshold_ratio*100}% Items": stats["js_div_frequent"] if stats else None,
                "All Items": stats["js_div_all"] if stats else None,
                "Real Null %": stats["real_null_total"] if stats else None,
                "Syn Null %": stats["syn_null_total"] if stats else None,
            })
            
        # Process numeric columns
        for col in self.numeric_columns:
            stats = get_ks_statistics(col, ks_pivot_df)
            summary_data.append({
                "Table": self.table_name,
                "Column": col,
                "Category": "Numeric",
                "Column-wise": stats["ks_stats_total"] if stats else None,
                f"Top {self.top_k} Items": stats["ks_stats_topk"] if stats else None,
                f"≥{self.freq_threshold_ratio*100}% Items": stats["ks_stats_frequent"] if stats else None,
                "All Items": stats["ks_stats_all"] if stats else None,
                "Real Null %": stats["real_null_total"] if stats else None,
                "Syn Null %": stats["syn_null_total"] if stats else None,
            })
    
        summary_df = pd.DataFrame(summary_data)

       # Move self.item_column-related rows to the top
        if self.item_column in summary_df["Column"].values:
            condition_rows = summary_df[summary_df["Column"] == self.item_column]
            other_rows = summary_df[summary_df["Column"] != self.item_column]
            summary_df = pd.concat([condition_rows, other_rows], axis=0).reset_index(drop=True)
    
        if include_mean:
            mean_row = pd.DataFrame(summary_df.mean(axis=0, numeric_only=True)).T.round(3)
            mean_row.index = ['Mean']
            summary_df = pd.concat([summary_df, mean_row], axis=0)
    
        return summary_df