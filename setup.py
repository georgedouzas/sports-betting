from setuptools import setup, find_packages

setup(
    name="sportsbet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'progressbar2',
        'scipy',
        'numpy',
        'pandas',
        'scikit-learn==0.19.2',
        'imbalanced-learn==0.3.3',
        'category_encoders',

    ]
)