"""
Конфигурационные параметры
"""
from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).parent

# Путь к исходным данным
DATA_ROOT = PROJECT_ROOT / "inria_aerial_dataset"

# Путь для различных результатов
RESULTS_PATH = PROJECT_ROOT / "results"

# Путь для сохранения обработанных снимков
PREPROCESSES_IMAGES_PATH = RESULTS_PATH / "AerialImageDataset_for_DINOv2"

# Пути для сохранения embedding
TRAIN_EMBEDDING_CITY_PATH = RESULTS_PATH / "embedding" / "train"
TEST_EMBEDDING_CITY_PATH = RESULTS_PATH / "embedding" / "test"

# Пути для сохранения разрезанных снимков
TRAIN_SLICED_IMAGES_PATH = RESULTS_PATH / "sliced_images" / "train"
TEST_SLICED_IMAGES_PATH = RESULTS_PATH / "sliced_images" / "test"

# Путь для сохранения обрезанных изображений
CROP_IMAGE_PATH = RESULTS_PATH / "AerialImageDataset_for_DINOv2"

# Путь для финальных данных городов 
CITY_DATA_PATH = RESULTS_PATH / "finish_city_data"

# Правильная последовательность снимков для городов
SEQUENCE = [6, 12, 18, 24, 30, 36,
            5, 11, 17, 23, 29, 35,
            4, 10, 16, 22, 28, 34,
            3, 9, 15, 21, 27, 33,
            2, 8, 14, 20, 26, 32,
            1, 7, 13, 19, 25, 31]

# Списки городов для обучения и валидации
TRAIN_CITY_LIST = ["austin"]
# ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
TEST_CITY_LIST = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]

# Размер изображения для DINOv2
DINO_IMG_SIZE = 518

# ImageNet нормализация
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Путь для данных нарезанных изображений
SLICED_CITY_DATA_PATH = RESULTS_PATH / "sliced_city_data"

# Пути для CSV с координатами городов
TRAIN_CITY_DATA_PATH = RESULTS_PATH / "cities_data" / "train"
TEST_CITY_DATA_PATH = RESULTS_PATH / "cities_data" / "test"