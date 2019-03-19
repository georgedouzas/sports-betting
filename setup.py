from setuptools import setup, find_packages

setup(
    name="sportsbet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'scipy==1.1.0',
        'numpy==1.16.2',
        'pandas==0.24.2',
        'scikit-learn==0.20.3',
        'imbalanced-learn==0.4.2',
        'tqdm==4.28.1'
    ],
    entry_points={
        'console_scripts':[
            'download=sportsbet.soccer.data:download',
            'backtest=sportsbet.soccer.optimization:backtest',
            'predict=sportsbet.soccer.optimization:predict'
        ]
    }
)
