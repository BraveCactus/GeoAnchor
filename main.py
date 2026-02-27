import os
import csv
import pandas as pd
from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import ImageFile, Image

from src.scripts.images_tools import view_geotiff

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from config import DATA_ROOT, RESULTS_PATH
from src.scripts.images_tools import view_geotiff
from src.scripts.preprocess_images import extract_embedding_from_image

from src.scripts.seasons_dataset import (
    create_train_season_data,
    create_test_season_data,
    create_all_seasons,
    stitch_austin_maps
)

from src.scripts.standart_dataset import (
    create_train_data,
    create_test_data,
    get_city_bounds
)

from src.scripts.extra_dataset import (
    create_chicago_dataset,
    get_chicago_bounds,
    create_rotated_patches_dataset,
    load_city_bounds
)

train_data_path = DATA_ROOT / "AerialImageDataset" / "train" / "images"
train_csv_path = RESULTS_PATH / "cities_data" / "train"
sliced_output_path = RESULTS_PATH / "sliced_city_data"
output_maps_path = RESULTS_PATH / "maps" / "final" 

df_path = RESULTS_PATH / "sliced_city_data" / "austin_sliced_with_coords.csv"
map_path = RESULTS_PATH / "maps" / "final" / "austin_map.tiff"

test_image_path = DATA_ROOT / "AerialImageDataset" / "train" / "images" / "austin1.tif"

if __name__ == "__main__":
    pass