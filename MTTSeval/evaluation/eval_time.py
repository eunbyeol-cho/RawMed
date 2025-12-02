import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings(action='ignore')

class TimeAnalyzer:
    def __init__(self, ehr, real_times, synthetic_times):
        self.ehr = ehr
        self.real_times = real_times
        self.synthetic_times = synthetic_times
        self.real_intervals = [np.diff(seq) for seq in real_times]
        self.synthetic_intervals = [np.diff(seq) for seq in synthetic_times]

    @staticmethod
    def calculate_median_iqr(data):
        """
        Calculate the median and interquartile range (IQR) of a dataset.
        """
        return np.median(data), np.percentile(data, 75) - np.percentile(data, 25)

    @staticmethod
    def perform_ks_test(real_data, synthetic_data, exclude_zeros=False):
        """
        Perform the Kolmogorov-Smirnov (KS) test between two datasets.
        """
        if exclude_zeros:
            real_data, synthetic_data = real_data[real_data > 0], synthetic_data[synthetic_data > 0]
        ks_stat, p_value = ks_2samp(real_data, synthetic_data)
        return ks_stat, p_value

    @staticmethod
    def plot_cdf(real_data, synthetic_data, label, exclude_zeros=False):
        """
        Plot the cumulative distribution function (CDF) for real and synthetic data.
        """
        if exclude_zeros:
            real_data = real_data[real_data > 0]
            synthetic_data = synthetic_data[synthetic_data > 0]

        plt.figure()
        sorted_real = np.sort(real_data)
        sorted_synthetic = np.sort(synthetic_data)
        yvals_real = np.arange(1, len(sorted_real) + 1) / float(len(sorted_real))
        yvals_synthetic = np.arange(1, len(sorted_synthetic) + 1) / float(len(sorted_synthetic))
        
        plt.plot(sorted_real, yvals_real, label=f'Real {label}')
        plt.plot(sorted_synthetic, yvals_synthetic, label=f'Synthetic {label}')
        plt.title(f'CDF of {label} {"(Excluding Zeros)" if exclude_zeros else ""}')
        plt.xlabel(label)
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_box(real_data, synthetic_data, label, exclude_zeros=False):
        """
        Plot a box plot for real and synthetic data.
        """
        if exclude_zeros:
            real_data = real_data[real_data > 0]
            synthetic_data = synthetic_data[synthetic_data > 0]

        plt.figure()
        plt.boxplot([real_data, synthetic_data], labels=[f'Real {label}', f'Synthetic {label}'])
        plt.title(f'Box Plot of {label} {"(Excluding Zeros)" if exclude_zeros else ""}')
        plt.ylabel(label)
        plt.grid(True)
        plt.show()

    def collect_statistics(self, real_data, synthetic_data, label, plot_type=None, exclude_zeros=False):
        """
        Collect statistics (median, IQR, KS test) and plot for a given dataset.
        """
        real_data_flattened = np.array(real_data)
        synthetic_data_flattened = np.array(synthetic_data)

        if plot_type == 'cdf':
            self.plot_cdf(real_data_flattened, synthetic_data_flattened, label, exclude_zeros)
        elif plot_type == 'box':
            self.plot_box(real_data_flattened, synthetic_data_flattened, label, exclude_zeros)

        real_median, real_iqr = self.calculate_median_iqr(real_data_flattened)
        synthetic_median, synthetic_iqr = self.calculate_median_iqr(synthetic_data_flattened)
        ks_stat, p_value = self.perform_ks_test(real_data_flattened, synthetic_data_flattened, exclude_zeros)

        return {
            'Label': label,
            'Real Median': real_median,
            'Synthetic Median': synthetic_median,
            'Real IQR': real_iqr,
            'Synthetic IQR': synthetic_iqr,
            'KS Stat': ks_stat,
            'p-value': p_value
        }

    def analyze_event_counts(self, plot_type=None):
        """
        Analyze event counts in real and synthetic datasets.
        """
        synthetic_event_counts = [len(seq) for seq in self.synthetic_times]
        real_event_counts = [len(seq) for seq in self.real_times]

        return self.collect_statistics(real_event_counts, synthetic_event_counts, 'Event Counts', plot_type)

    def analyze(self, plot_type=None):
        """
        Perform the complete time analysis, including time intervals, absolute time, and event counts.
        """
        results = []

        # Time Intervals (differences between events)
        real_intervals_flattened = np.concatenate(self.real_intervals)
        synthetic_intervals_flattened = np.concatenate(self.synthetic_intervals)

        interval_stats_with_zeros = self.collect_statistics(
            real_intervals_flattened, synthetic_intervals_flattened,
            'Time Intervals (Including Zeros)', plot_type
        )

        interval_stats_without_zeros = self.collect_statistics(
            real_intervals_flattened[real_intervals_flattened > 0],
            synthetic_intervals_flattened[synthetic_intervals_flattened > 0],
            'Time Intervals (Excluding Zeros)', plot_type
        )

        results.extend([interval_stats_with_zeros, interval_stats_without_zeros])

        # Absolute Time (raw time values)
        real_absolute_time_flattened = np.concatenate(self.real_times)
        synthetic_absolute_time_flattened = np.concatenate(self.synthetic_times)

        absolute_time_stats = self.collect_statistics(
            real_absolute_time_flattened, synthetic_absolute_time_flattened,
            'Absolute Time', plot_type
        )
        results.append(absolute_time_stats)

        # Event Counts
        event_count_stats = self.analyze_event_counts(plot_type)
        results.append(event_count_stats)

        # Convert results to a DataFrame for better readability
        results_df = pd.DataFrame(results)
        return results_df