import math, os
import numpy as np
import pandas as pd

from typing import Dict

class DataLoader():
    def __init__(self, data_path: str, parse_func: callable = None):
        self._validate_file_path(data_path)    
        self.data_path = data_path
        # Add logic to apply parsing function to data (transform it into a row with query + gold)
        self.data = self._load_data()

    def get(self, index: int = None) -> Dict:
        """Get query, gold pair (+ extra data) at index (or random index if None)"""
        if index is None:
            index = np.random.randint(0, len(self.data))
        record = self.data.iloc[index].to_dict()
        record = {
            key: value for key, value in record.items()
            if not (isinstance(value, float) and math.isnan(value))
        }
        return record

    def _load_data(self):
        """Load data from file path into pandas dataframe"""
        file_ext = self.data_path.split(".")[-1]
        if file_ext == "csv":
            data = pd.read_csv(self.data_path)
        elif file_ext == "tsv":
            data = pd.read_csv(self.data_path, sep="\t")
        elif file_ext == "json":
            data = pd.read_json(self.data_path)
        elif file_ext == "jsonl":  # Added support for JSON Lines
            data = pd.read_json(self.data_path, lines=True)
        elif file_ext == "pickle" or file_ext == "pkl":
            data = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # if "query" not in data.columns:
            # raise ValueError("Data must have 'query' column/field")
        return data
    
    def _validate_file_path(self, file_path: str):
        """Check if the file extension is one of tsv, csv, json, jsonl, or pickle"""
        if not os.path.exists(file_path):
            raise OSError(f"Invalid file path: {file_path}")
        valid_extensions = [".tsv", ".csv", ".pickle", ".pkl", ".json", ".jsonl"]  # Added .jsonl
        _, ext = os.path.splitext(file_path)
        if ext not in valid_extensions:
            raise ValueError(f"File type is not supported: {file_path}")

    def __len__(self):
        return len(self.data)