from setuptools import setup, find_packages

setup(
    name="logistic_classifier",
    version="0.1",
    packages=find_packages(where='src'),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
    ],
)