import os
import json
import logging
import pandas as pd


def load_data_from_folder(folder_path):
    logging.info(f"Loading data from folder: {folder_path}")
    datasets = []
    headers = []

    for folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder)
        if os.path.isdir(folder_path_full):
            logging.info(f"Processing folder: {folder_path_full}")
            for file in os.listdir(folder_path_full):
                if file.endswith(".data"):
                    file_path = os.path.join(folder_path_full, file)
                    df = pd.read_csv(file_path, delimiter=',')
                    headers.append((file, df.columns.tolist()))
                    datasets.append((file, df))
                    logging.info(f"Loaded dataset: {file_path}, shape: {df.shape}")

    logging.info(f"Finished loading datasets. Total datasets loaded: {len(datasets)}")
    return datasets, headers


def load_metadata(folder_path, dataset_name):
    metadata_path = os.path.join(folder_path, f"{dataset_name}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
            logging.info(f"Metadata loaded for {dataset_name}: {metadata}")
            return metadata
    else:
        logging.info(f"No metadata found for {dataset_name}")
        return {}