import json
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
import random
import pyproj
from PIL import Image
from torchvision import transforms

from src.models.models import Dinov2EmbendingExtractor
from src.scripts.images_tools import utm26910_to_wgs84 

def extract_random_patches(image_path, output_path, num_patches=100, patch_size=518):
    """Извлекает случайные патчи из большого изображения и сохраняет их в указанную папку
        Args:        
            image_path (str): Путь к большому изображению
            output_path (Path): Папка для сохранения патчей
            num_patches (int): Количество патчей для извлечения
            patch_size (int): Размер квадратного патча (по умолчанию 518x518)
        Returns:        
            pathes: Список словарей с информацией о каждом патче (имя, путь, координаты в большом изображении)
            h, w: Высота и ширина большого изображения в пикселях
    """
    img = tiff.imread(image_path)
    h, w = img.shape[:2]
    
    patches = []
    
    for i in range(num_patches):
        y = random.randint(0, h - patch_size)
        x = random.randint(0, w - patch_size)
        
        patch = img[y:y+patch_size, x:x+patch_size]
        name = f"test_patch_{i:03d}.tiff"
        path = output_path / name
        tiff.imwrite(str(path), patch)
        
        patches.append({
            'name': name,
            'path': path,
            'x': x,
            'y': y
        })
    
    return patches, w, h

def extract_embedding(img_path, extractor):
    """Извлекает эмбеддинг из изображения с помощью модели DINOv2
        Args:
            img_path (str): Путь к изображению
            extractor (Dinov2EmbendingExtractor): Экстрактор эмбеддингов DINOv2
        Returns:
            str: Эмбеддинг в виде строки, разделенной точкой с запятой, или пустая строка при ошибке
    """
    try:
        img = Image.open(img_path)
        transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0)
        emb = extractor.extract_embedding(tensor)
        return ';'.join(map(str, emb))
    except:
        return ""   


def create_test_dataset(season='autumn', num_patches=100, extractor=Dinov2EmbendingExtractor(), in_path=None, out_path=None):
    """Создает тестовый датасет для указанного сезона, извлекая случайные патчи из большого изображения и сохраняя их вместе с метаданными в CSV файл
        Args:
            season (str): Название сезона
            num_patches (int): Количество патчей для извлечения
            extractor (Dinov2EmbendingExtractor): Экстрактор эмбеддингов DINOv2
            in_path (str): Путь к папке с большим изображениями сезонов
            out_path (str): Путь к папке для сохранения тестового датасета
        Returns:
            pd.DataFrame: DataFrame с метаданными и эмбеддингами для каждого патча
    """
    print(f"\nСоздание тестового датасета для {season}")
    
    source_path = Path(in_path) / f"austin_{season}_full.tiff"
    target_path = Path(out_path) / season

    out_path = target_path / "images"
    csv_path = target_path / "data"
    
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path.mkdir(parents=True, exist_ok=True)
    
    patches, img_w, img_h = extract_random_patches(source_path, out_path, num_patches)
    
    with open("src/data/city_data.json", "r") as f:
        city_data = json.load(f)
        
    map_left = city_data["map_left"]
    map_bottom = city_data["map_bottom"]
    map_right = city_data["map_right"]
    map_top = city_data["map_top"]

    records = []
    
    for i, p in enumerate(patches):
        emb = extract_embedding(p['path'], extractor)
        
        px = p['x'] + 259
        py = p['y'] + 259
        
        meter_x = (map_right - map_left) / img_w
        meter_y = (map_top - map_bottom) / img_h
        
        cx = map_left + px * meter_x
        cy = map_top - py * meter_y
        
        lat, lon = utm26910_to_wgs84(cx, cy)
        
        rel_x = (cx - map_left) / (map_right - map_left)
        rel_y = (cy - map_bottom) / (map_top - map_bottom)
        
        records.append({
            'patch_id': i,
            'patch_name': p['name'],
            'season': season,
            'tile_path': str(p['path']),
            'lat': lat,
            'lon': lon,
            'center_x_utm': cx,
            'center_y_utm': cy,
            'rel_x': rel_x,
            'rel_y': rel_y,
            'split': 'test',
            'embedding': emb
        })
        
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{num_patches}")
    
    df = pd.DataFrame(records)
    csv_file = csv_path / f"test_{season}_patches.csv"
    df.to_csv(csv_file, index=False)
    print(f"Сохранено {len(df)} записей в {csv_file}")
    
    return df

def create_test_season_data(seasons=['autumn'], num_patches=100, extractor=None, in_path=None, out_path=None):
    """Создает тестовые датасеты для всех указанных сезонов
        Args:
            seasons (list): Список сезонов для создания датасетов
            num_patches (int): Количество патчей для извлечения для каждого сезона
            extractor (Dinov2EmbendingExtractor): Экстрактор эмбеддингов DINOv2
            in_path (str): Путь к папке с большим изображениями сезонов
            out_path (str): Путь к папке для сохранения тестовых датасетов
        Returns:
            None
    """
    print("="*50)
    print("СОЗДАНИЕ ТЕСТОВЫХ ДАТАСЕТОВ")
    print("="*50)    

    for season in seasons:
        create_test_dataset(season, num_patches, extractor, in_path, out_path)
    
    print("\nГотово!")

if __name__ == "__main__":
    in_path = Path("results/stitched_seasons_austin")
    out_path = Path("results/test_dataset")

    extractor=Dinov2EmbendingExtractor()

    seasons = ['autumn', 'winter']
    
    create_test_season_data(seasons=seasons, num_patches=600, extractor=extractor, in_path=in_path, out_path=out_path)