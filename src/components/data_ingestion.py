from src.config import RAW_DATA_DIR
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import zipfile
import os

load_dotenv()

COMPETITION_NAME = "playground-series-s6e2"

def run_ingestion():
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(RAW_DATA_DIR):
        os.mkdir(RAW_DATA_DIR)
        print("Created directory for data ingestions...")
    
    api.competition_download_files(competition=COMPETITION_NAME, path=RAW_DATA_DIR)

    zip_file = f"{RAW_DATA_DIR}/{COMPETITION_NAME}.zip"
    
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(path=RAW_DATA_DIR)

        os.remove(zip_file)
        print("Zip File extracted successfully...")
    else:
        print("Zip File Not Found...")
    
if __name__ == "__main__":
    run_ingestion()