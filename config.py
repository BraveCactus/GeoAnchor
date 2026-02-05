"""
Конфигурационные параметры
"""
from pathlib import Path

# 1. БАЗОВЫЕ ПУТИ
PROJECT_ROOT = Path(__file__).parent

# Путь к исходным данным
DATA_ROOT = PROJECT_ROOT / "inria_aerial_dataset"

# Путь для сохранения результатов
PREPROCESSES_IMAGES_ROOT = DATA_ROOT / "AerialImageDataset_for_DINOv2"