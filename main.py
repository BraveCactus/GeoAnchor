import os
import pandas as pd
from pathlib import Path

from config import DATA_ROOT, RESULTS_PATH, TRAIN_CITY_LIST, TEST_CITY_LIST
from src.data.preprocess_images import load_and_preprocess_tiff, extract_embedding_from_tiff
from src.data.images_tools import read_tiff_metadata, build_city_compressed_map, get_city_data

train_data_path = DATA_ROOT / r"AerialImageDataset\train\images"
test_data_path = DATA_ROOT / r"AerialImageDataset\test\images"


map_folder_path = RESULTS_PATH / "maps"
train_map_folder_path = RESULTS_PATH / "maps" / "train"
test_map_folder_path = RESULTS_PATH / "maps" / "test"


train_csv_city_data_path = RESULTS_PATH / "cities_data" / "train"
test_csv_city_data_path = RESULTS_PATH / "cities_data" / "test"

test_img_path = train_data_path / "austin1.tif"


if __name__ == "__main__":  

    # Извлечение данных о городах
    # for city in TRAIN_CITY_LIST:
    #     get_city_data(train_data_path, train_csv_city_data_path, city)

    # for city in TEST_CITY_LIST:
    #     get_city_data(test_data_path, test_csv_city_data_path, city)

    # Построение полных карт городов
    # for city in TRAIN_CITY_LIST:
    #     build_city_compressed_map(train_data_path, train_map_folder_path, city)

    # for city in TEST_CITY_LIST:
    #     build_city_compressed_map(test_data_path, test_map_folder_path, city)

    city_path = train_map_folder_path = RESULTS_PATH / "cities_data" / "train" / "kitsap_data.csv"
    city_data = pd.read_csv(city_path)

    # min_left = city_data.loc[city_data["left"].idxmin()]["left"]  
    # min_bottom = city_data.loc[city_data["bottom"].idxmin()]["bottom"]

    # max_top = city_data.loc[city_data["top"].idxmax()]["top"] 
    # max_right = city_data.loc[city_data["right"].idxmax()]["right"]

    # print(city_data.loc[(city_data["left"] == min_left) & (city_data["bottom"] == min_bottom)]["image name"])
    # print(city_data.loc[(city_data["top"] == max_top) & (city_data["right"] == max_right)]["image name"])

    # Загрузка данных
    # df = pd.read_csv(city_path)

    # 1. Находим минимальное значение bottom среди ВСЕХ снимков
    # min_bottom_value = df['bottom'].min()
    # print(f"Минимальное значение bottom среди всех снимков: {min_bottom_value}")

    # 2. Находим ВСЕ снимки с этим значением
    # all_min_bottom = df[df['bottom'] == min_bottom_value]

    # print(f"\nНайдено {len(all_min_bottom)} снимков с минимальным bottom:")
    # for idx, row in all_min_bottom.iterrows():
    #     print(f"{row['image name']}: bottom={row['bottom']}, left={row['left']}")
    
    embedding = extract_embedding_from_tiff(test_img_path)