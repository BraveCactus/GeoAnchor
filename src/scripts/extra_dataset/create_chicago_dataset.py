# create_chicago_dataset.py
import numpy as np
import tifffile as tiff
from pathlib import Path
import pandas as pd
import rasterio
from torchvision import transforms
from PIL import Image

from config import DATA_ROOT, RESULTS_PATH
from src.scripts.images_tools import build_city_map, utm26910_to_wgs84, read_tiff_metadata
from src.models.models import Dinov2EmbendingExtractor

def get_chicago_bounds(data_path, city="chicago"):
    """Получает границы города из всех снимков"""
    city_files = list(data_path.glob(f"{city}*.tif"))
    
    if not city_files:
        return None, None, None, None, None
    
    first_metadata = read_tiff_metadata(city_files[0])
    map_left = float('inf')
    map_bottom = float('inf')
    map_right = float('-inf')
    map_top = float('-inf')
    crs = first_metadata['crs']
    
    for img_file in city_files:
        metadata = read_tiff_metadata(img_file)
        left, bottom, right, top = metadata["bounds"]
        
        map_left = min(map_left, left)
        map_bottom = min(map_bottom, bottom)
        map_right = max(map_right, right)
        map_top = max(map_top, top)
    
    print(f"Границы города {city}:")
    print(f"  left={map_left:.1f}, bottom={map_bottom:.1f}")
    print(f"  right={map_right:.1f}, top={map_top:.1f}")
    
    return map_left, map_bottom, map_right, map_top, crs

def extract_random_patches(map_file, output_path, num_patches=600, patch_size=518):
    """Вырезает случайные патчи 518x518 из сшитой карты города"""
    
    city_image = tiff.imread(map_file)
    
    if len(city_image.shape) == 2:
        city_image = np.stack([city_image, city_image, city_image], axis=2)
    
    height, width = city_image.shape[:2]
    
    patches_info = []
    
    print(f"Вырезание {num_patches} патчей...")
    
    for patch_num in range(num_patches):
        max_y = height - patch_size
        max_x = width - patch_size
        
        y_start = np.random.randint(0, max_y)
        x_start = np.random.randint(0, max_x)
        y_end = y_start + patch_size
        x_end = x_start + patch_size
        
        patch = city_image[y_start:y_end, x_start:x_end]
        
        patch_name = f"chicago_patch_{patch_num:03d}.tiff"
        patch_path = output_path / patch_name
        tiff.imwrite(str(patch_path), patch)
        
        patches_info.append({
            'patch_id': patch_num,
            'patch_name': patch_name,
            'grid_i': y_start,
            'grid_j': x_start,
            'y_start': y_start,
            'x_start': x_start,
            'patch_path': patch_path
        })
        
        if patch_num % 100 == 0:
            print(f"  Создано {patch_num}/{num_patches} патчей")
    
    print(f"Всего создано {len(patches_info)} патчей")
    return patches_info

def get_relative_coords(map_left, map_bottom, map_right, map_top, 
                       patch_left, patch_bottom, patch_right, patch_top):
    """Относительные координаты центра патча (0-1)"""
    center_x = (patch_left + patch_right) / 2
    center_y = (patch_bottom + patch_top) / 2
    
    width = map_right - map_left
    height = map_top - map_bottom
    
    rel_x = (center_x - map_left) / width
    rel_y = (center_y - map_bottom) / height
    
    rel_x = max(0.0, min(1.0, rel_x))
    rel_y = max(0.0, min(1.0, rel_y))
    
    return rel_x, rel_y

def extract_embeddings_for_patches(patches_info):
    """Извлекает эмбеддинги для патчей"""
    
    print("Извлечение эмбеддингов...")
    extractor = Dinov2EmbendingExtractor()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    embeddings_dict = {}
    
    for idx, patch_info in enumerate(patches_info):
        if idx % 50 == 0:
            print(f"  Обработано {idx}/{len(patches_info)} патчей...")
        
        try:
            patch_path = patch_info['patch_path']
            img_data = tiff.imread(patch_path)
            
            if len(img_data.shape) == 2:
                img_data = np.stack([img_data, img_data, img_data], axis=2)
            
            pil_img = Image.fromarray(img_data)
            
            if pil_img.size != (518, 518):
                pil_img = pil_img.resize((518, 518), Image.Resampling.LANCZOS)
            
            img_tensor = transform(pil_img).unsqueeze(0)
            embedding = extractor.extract_embedding(img_tensor)
            embedding_str = ';'.join(map(str, embedding))
            
            embeddings_dict[patch_info['patch_name']] = embedding_str
            
        except Exception:
            embeddings_dict[patch_info['patch_name']] = ""
    
    return embeddings_dict

def process_chicago_patches(data_path, patches_info, output_csv_path):
    """Создает CSV с координатами и эмбеддингами патчей Chicago"""
    
    map_left, map_bottom, map_right, map_top, crs = get_chicago_bounds(data_path, "chicago")
    
    if None in [map_left, map_bottom, map_right, map_top]:
        print("Ошибка: не удалось получить границы города")
        return
    
    print(f"Границы Chicago для расчетов:")
    print(f"  left={map_left:.1f}, bottom={map_bottom:.1f}")
    print(f"  right={map_right:.1f}, top={map_top:.1f}")
    
    embeddings_dict = extract_embeddings_for_patches(patches_info)
    
    records = []    

    total_width = 30000
    total_height = 30000
    
    pixel_to_meter_x = (map_right - map_left) / total_width
    pixel_to_meter_y = (map_top - map_bottom) / total_height
    
    for patch_info in patches_info:
        patch_name = patch_info['patch_name']
        patch_id = patch_info['patch_id']
        y_start = patch_info['y_start']
        x_start = patch_info['x_start']        
        
        patch_left = map_left + x_start * pixel_to_meter_x
        patch_right = patch_left + 518 * pixel_to_meter_x
        patch_top = map_top - y_start * pixel_to_meter_y
        patch_bottom = patch_top - 518 * pixel_to_meter_y        
        
        if patch_bottom > patch_top:
            patch_bottom, patch_top = patch_top, patch_bottom
        
        center_x = (patch_left + patch_right) / 2
        center_y = (patch_bottom + patch_top) / 2
        
        rel_x, rel_y = get_relative_coords(
            map_left, map_bottom, map_right, map_top,
            patch_left, patch_bottom, patch_right, patch_top
        )        
        
        lat, lon = utm26910_to_wgs84(center_x, center_y, "EPSG:26916")
        
        embedding = embeddings_dict.get(patch_name, "")
        
        record = {
            "patch_id": patch_id,
            "patch_name": patch_name,
            "original_image": "chicago_map",
            "patch_coords": f"{y_start}_{x_start}",
            "tile_path": str(patch_info['patch_path']),
            "lat": lat,
            "lon": lon,
            "patch_left": patch_left,
            "patch_bottom": patch_bottom,
            "patch_right": patch_right,
            "patch_top": patch_top,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "split": "test",
            "embedding": embedding
        }
        
        records.append(record)
    
    if records:
        df = pd.DataFrame(records)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Создан CSV: {output_csv_path} ({len(df)} записей)")
        
        print("\nСтатистика координат Chicago:")
        print(f"Диапазон широт: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
        print(f"Диапазон долгот: [{df['lon'].min():.6f}, {df['lon'].max():.6f}]")
    
    return df

def create_chicago_dataset():
    """Создает датасет для города Chicago (600 изображений)"""
    
    data_path = DATA_ROOT / "AerialImageDataset" / "train" / "images"
    
    chicago_path = RESULTS_PATH / "chicago"
    chicago_maps_path = chicago_path / "maps"
    chicago_patches_path = chicago_path / "patches"
    chicago_data_path = chicago_path / "data"
    
    for path in [chicago_path, chicago_maps_path, chicago_patches_path, chicago_data_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    print("Создание датасета для города Chicago")
    print("=" * 50)    
    
    print("1. Склеиваем карту Chicago из тайлов")
    build_city_map(data_path, chicago_maps_path, "chicago")    
    
    map_file = chicago_maps_path / "chicago_map.tiff"
    print(f"2. Вырезаем 600 патчей из {map_file}")
    patches_info = extract_random_patches(map_file, chicago_patches_path, num_patches=600)    
    
    print("3. Создаем CSV с координатами и эмбеддингами")
    csv_path = chicago_data_path / "chicago_patches.csv"
    df = process_chicago_patches(data_path, patches_info, csv_path)
    
    print("\nДатасет Chicago создан")
    print(f"Патчи: {chicago_patches_path}")
    print(f"CSV: {csv_path}")
    
    return csv_path

if __name__ == "__main__":
    create_chicago_dataset()