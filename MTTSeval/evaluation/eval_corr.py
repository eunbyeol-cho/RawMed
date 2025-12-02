import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from dython.nominal import correlation_ratio, theils_u
import warnings
warnings.filterwarnings(action='ignore')

class CorrelationAnalyzer:
    def __init__(self, config, table_name, numeric_columns, categorical_columns):
        self.table_name = table_name
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.item_column = config["item_column"][table_name]
        self.pid_col = config["pid_column"]
        self.time_col = config["time_column"]

    def calculate_correlation_matrix(self, df):
        """
        Calculate the correlation matrix for the given DataFrame.
        """
        df = df.drop(columns=[col for col in [self.pid_col, self.time_col] if col in df.columns], errors="ignore")
        columns = self.numeric_columns + self.categorical_columns
        n = len(columns)
        corr_matrix = np.zeros((n, n))

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    corr_matrix[i, j] = 1
                elif i < j:
                    temp_df = df[[col1, col2]].dropna()
                    corr = self.calculate_pairwise_correlation(temp_df, col1, col2)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(corr_matrix, index=columns, columns=columns)

    def calculate_pairwise_correlation(self, temp_df, col1, col2):
        """
        Calculate pairwise correlation for two columns based on their types.
        """
        if temp_df.empty:
            return 0
        if col1 in self.numeric_columns and col2 in self.numeric_columns:
            return temp_df[col1].corr(temp_df[col2], method="pearson")
        if col1 in self.categorical_columns and col2 in self.numeric_columns:
            return correlation_ratio(temp_df[col1], temp_df[col2])
        if col1 in self.numeric_columns and col2 in self.categorical_columns:
            return correlation_ratio(temp_df[col2], temp_df[col1])
        return theils_u(temp_df[col1], temp_df[col2])

    @staticmethod
    def calculate_mu_absolute(real_corr_matrix, synthetic_corr_matrix):
        """
        Calculate the mean absolute difference between two correlation matrices.
        """
        return np.mean(np.abs(real_corr_matrix - synthetic_corr_matrix))

    @staticmethod
    def calculate_correlation_accuracy(real_corr_matrix, synthetic_corr_matrix):
        """
        Calculate the correlation accuracy between two correlation matrices.
        """
        def discretize_correlation_value(correlation):
            if correlation < -0.5:
                return 0  # Strong negative
            elif -0.5 <= correlation < -0.3:
                return 1  # Medium negative
            elif -0.3 <= correlation < -0.1:
                return 2  # Weak negative
            elif -0.1 <= correlation <= 0.1:
                return 3  # None
            elif 0.1 < correlation <= 0.3:
                return 4  # Weak positive
            elif 0.3 < correlation <= 0.5:
                return 5  # Medium positive
            else:
                return 6  # Strong positive

        real_discrete = np.vectorize(discretize_correlation_value)(real_corr_matrix)
        synthetic_discrete = np.vectorize(discretize_correlation_value)(synthetic_corr_matrix)
        return np.mean(real_discrete == synthetic_discrete)

    def analyze(self, real_df, synthetic_df, plot=False):
        """
        Perform correlation analysis on real and synthetic data.
        """
        # Calculate correlation matrices
        real_corr_matrix = self.calculate_correlation_matrix(real_df)
        synthetic_corr_matrix = self.calculate_correlation_matrix(synthetic_df)

        # Align matrices
        column_order = real_df.columns.difference([self.pid_col, self.time_col])
        real_corr_matrix = real_corr_matrix.loc[column_order, column_order]
        synthetic_corr_matrix = synthetic_corr_matrix.loc[column_order, column_order]

        # Calculate metrics
        mu_abs = self.calculate_mu_absolute(real_corr_matrix, synthetic_corr_matrix)
        cor_acc = self.calculate_correlation_accuracy(real_corr_matrix, synthetic_corr_matrix)

        # Plot matrices if required
        if plot:
            self.plot_correlation_matrices(real_corr_matrix, synthetic_corr_matrix, mu_abs, cor_acc)
            self.plot_difference_matrix(real_corr_matrix, synthetic_corr_matrix, mu_abs, cor_acc)

        return {
            'real_correlation_matrix': real_corr_matrix,
            'synthetic_correlation_matrix': synthetic_corr_matrix,
            'mu_abs': round(mu_abs, 3),
            'cor_acc': round(cor_acc, 3),
        }

    def analyze_by_item(self, real_df, synthetic_df, plot=False):
        """
        Perform item-specific correlation analysis.
        """
        results = {}
        total_rows = len(real_df)  # Total number of rows in real_df
        threshold = total_rows * 0.001  # 0.1% of total rows    
        
        for item in tqdm(real_df[self.item_column].unique(), desc="Item-Specific Analysis"):
            # Filter data for the current item
            real_item_df = real_df[real_df[self.item_column] == item]
            synthetic_item_df = synthetic_df[synthetic_df[self.item_column] == item]

            # Skip if not enough data or less than 0.1% of total rows
            if len(real_item_df) < 100 or len(synthetic_item_df) < 100:
                continue
            
            # Perform analysis for the item
            result = self.analyze(real_item_df, synthetic_item_df, plot=plot)
            results[item] = result

        return results

    def plot_correlation_matrices(self, real_corr_matrix, synthetic_corr_matrix, mu_abs, cor_acc, annotate=False):
        """
        Plot real and synthetic correlation matrices side by side.
        """
        plt.figure(figsize=(12, 6))

        # Real matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(real_corr_matrix, annot=annotate, fmt=".2f", cmap="coolwarm", square=True, vmin=-1, vmax=1)
        plt.title("Real Correlation Matrix")

        # Synthetic matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(synthetic_corr_matrix, annot=annotate, fmt=".2f", cmap="coolwarm", square=True, vmin=-1, vmax=1)
        plt.title("Synthetic Correlation Matrix")

        plt.suptitle(f"μ_abs: {mu_abs:.4f}, CorAcc: {cor_acc:.4f}", fontsize=16)
        plt.tight_layout()
        plt.show()
        
    def plot_difference_matrix(self, real_corr_matrix, synthetic_corr_matrix, mu_abs, cor_acc, annotate=False):
        """
        Plot the absolute difference between real and synthetic correlation matrices.
        """
        # Compute the absolute difference matrix
        diff_matrix = np.abs(real_corr_matrix - synthetic_corr_matrix)
    
        # Plot the difference matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(diff_matrix, annot=annotate, fmt=".2f", cmap="Greys", square=True, cbar=True, vmin=0, vmax=2)
        plt.title(f"Absolute Difference Matrix\nμ_abs: {mu_abs:.4f}, CorAcc: {cor_acc:.4f}", fontsize=16)
        plt.tight_layout()
        plt.show()

