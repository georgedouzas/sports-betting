from setuptools import setup, find_packages

setup(
    name="sportsbet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'scikit-learn'
    ]
)