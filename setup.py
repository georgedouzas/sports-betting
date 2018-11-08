from setuptools import setup, find_packages

setup(
    name="sportsbet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'scipy==1.1.0',
        'numpy==1.15.3',
        'pandas==0.23.4',
        'scikit-learn==0.20',
        'imbalanced-learn==0.4.2',
        'tqdm==4.28.1'
    ],
    scripts=[
        'sportsbet/scripts/fetch_data',
        'sportsbet/scripts/backtest_classifier',
        'sportsbet/scripts/evaluate_classifier',
        'sportsbet/scripts/fit_dump_classifier'
    ]
)
