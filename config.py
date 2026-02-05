"""
Конфигурационные параметры
"""
from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).parent

# Путь к исходным данным
DATA_ROOT = PROJECT_ROOT / "inria_aerial_dataset"

# Путь для сохранения результатов
PREPROCESSES_IMAGES_ROOT = DATA_ROOT / "AerialImageDataset_for_DINOv2"

# Путь для различных результатов
RESULTS_PATH = PROJECT_ROOT / "results"

# Правильная последовательность снимков для городов (из верхнего левого в правый нижний, слева направо)
SEQUENCE = [6, 12, 18, 24, 30, 36,
            5, 11, 17, 23, 29, 35,
            4, 10, 16, 22, 28, 34,
            3, 9, 15, 21, 27, 33,
            2, 8, 14, 20, 26, 32,
            1, 7, 13, 19, 25, 31,]

# Списки городов для обучения и валидации
TRAIN_CITY_LIST = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
TEST_CITY_LIST = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]


