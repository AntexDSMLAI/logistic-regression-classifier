# src/data_loader.py

import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file and return a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
     data = pd.read_csv(file_path)
     return data
    except FileNotFoundError:
       print(f"File {file_path}not found please check the file")
       return None

