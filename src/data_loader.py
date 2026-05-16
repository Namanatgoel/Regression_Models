import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise