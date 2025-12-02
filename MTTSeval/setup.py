from setuptools import setup, find_packages

setup(
    name="MTTSeval",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "tqdm",
        "xgboost",
        "scikit-learn",
    ],
) 