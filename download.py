import os
from dotenv import load_dotenv
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Load environment variables from .env (this includes KAGGLE_API_TOKEN)
load_dotenv()

# Load the dataset
file_path = "HI-Small_Trans.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
    file_path,
)

# Create data/raw directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save the dataset to data/raw/HI-Small_Trans.csv
df.to_csv("data/raw/HI-Small_Trans.csv", index=False)

print("Dataset downloaded and saved to data/raw/HI-Small_Trans.csv")
print("First 5 records:", df.head())