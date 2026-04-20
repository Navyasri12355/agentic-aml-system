import pytest, pandas as pd
from src.pipeline.data_ingestion import load_and_clean

def test_load_and_clean_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_clean("nonexistent.csv")