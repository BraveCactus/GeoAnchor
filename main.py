import os
import pandas as pd
from pathlib import Path

from config import DATA_ROOT, RESULTS_PATH, TRAIN_CITY_LIST, TEST_CITY_LIST
from src.data.preprocess_images import load_and_preprocess_tiff
from src.data.images_tools import read_tiff_metadata, build_city_compressed_map, get_city_data

train_data_path = DATA_ROOT / r"AerialImageDataset\train\images"
test_data_path = DATA_ROOT / r"AerialImageDataset\test\images"


map_folder_path = RESULTS_PATH / "maps"
train_map_folder_path = RESULTS_PATH / "maps" / "train"
test_map_folder_path = RESULTS_PATH / "maps" / "test"


train_csv_city_data_path = RESULTS_PATH / "cities_data" / "train"
test_csv_city_data_path = RESULTS_PATH / "cities_data" / "test"


if __name__ == "__main__":  

    # Извлечение данных о городах
    # for city in TRAIN_CITY_LIST:
    #     get_city_data(train_data_path, train_csv_city_data_path, city)

    # for city in TEST_CITY_LIST:
    #     get_city_data(test_data_path, test_csv_city_data_path, city)

    # Построение полных карт городов
    for city in TRAIN_CITY_LIST:
        build_city_compressed_map(train_data_path, train_map_folder_path, city)

    for city in TEST_CITY_LIST:
        build_city_compressed_map(test_data_path, test_map_folder_path, city)