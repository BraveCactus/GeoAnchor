# create_dataset.py
import numpy as np
import tifffile as tiff
from pathlib import Path
import pandas as pd
import rasterio
from torchvision import transforms
from PIL import Image

from config import DATA_ROOT, RESULTS_PATH
from src.scripts.images_tools import build_city_map, build_city_compressed_map, utm26910_to_wgs84, read_tiff_metadata
from src.models.models import Dinov2EmbendingExtractor

def get_city_bounds(data_path, city="austin"):
    """Получает границы города из всех снимков"""
    city_files = list(data_path.glob(f"{city}*.tif"))
    if not city_files:
        return None, None, None, None, None
    
    # Инициализируем значения
    first_metadata = read_tiff_metadata(city_files[0])
    map_left = float('inf')
    map_bottom = float('inf')
    map_right = float('-inf')
    map_top = float('-inf')
    crs = first_metadata['crs']
    
    # Находим минимальные и максимальные границы
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
    print(f"  Размер: {map_right-map_left:.1f}x{map_top-map_bottom:.1f} метров")
    
    return map_left, map_bottom, map_right, map_top, crs

def slice_city_map(map_path, output_path, patch_size=518):
    """Нарезает сшитую карту города 30000x30000 на патчи 518x518"""
    print("Загрузка сшитой карты...")
    
    map_file = map_path / "austin_map.tiff"
    city_image = tiff.imread(map_file)
    
    print(f"Размер карты: {city_image.shape}")
    
    num_patches = 30000 // patch_size
    
    patches_info = []
    
    print(f"Нарезка на {num_patches}x{num_patches} патчей...")
    
    patch_counter = 0
    
    for i in range(num_patches):
        for j in range(num_patches):
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size
            
            patch = city_image[y_start:y_end, x_start:x_end]
            
            patch_name = f"austin_patch_{i}_{j}.tiff"
            patch_path = output_path / patch_name
            tiff.imwrite(str(patch_path), patch)
            
            patches_info.append({
                'patch_id': patch_counter,
                'patch_name': patch_name,
                'grid_i': i,
                'grid_j': j,
                'y_start': y_start,
                'x_start': x_start,
                'patch_path': patch_path
            })
            
            patch_counter += 1
    
    print(f"Создано {len(patches_info)} патчей")
    return num_patches, patches_info

def stitch_patches_to_city(patches_info, output_path, num_patches, patch_size=518):
    """Сшивает патчи обратно в карту города"""
    print("Сшивание патчей...")
    
    city_size = num_patches * patch_size
    city_image = np.zeros((city_size, city_size, 3), dtype=np.uint8)
    
    for patch_info in patches_info:
        i = patch_info['grid_i']
        j = patch_info['grid_j']
        patch_path = patch_info['patch_path']
        
        patch = tiff.imread(patch_path)
        
        y_start = i * patch_size
        y_end = y_start + patch_size
        x_start = j * patch_size
        x_end = x_start + patch_size
        
        city_image[y_start:y_end, x_start:x_end] = patch
    
    output_file = output_path / "austin_stitched_from_patches.tiff"
    tiff.imwrite(str(output_file), city_image, compression='lzw')
    print(f"Сшитая карта сохранена: {output_file}")
    
    return city_image

def get_relative_coords(map_left, map_bottom, map_right, map_top, 
                       patch_left, patch_bottom, patch_right, patch_top):
    """Преобразует абсолютные координаты патча в относительные (0-1)"""
    patch_center_x = (patch_left + patch_right) / 2
    patch_center_y = (patch_bottom + patch_top) / 2
    
    width = map_right - map_left
    height = map_top - map_bottom
    
    rel_x = (patch_center_x - map_left) / width
    rel_y = (patch_center_y - map_bottom) / height
    
    return rel_x, rel_y

def extract_embeddings_for_patches(patches_info, batch_size=32):
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
        if idx % 100 == 0:
            print(f"Обработано {idx}/{len(patches_info)} патчей...")
        
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
            
        except Exception as e:
            print(f"Ошибка при обработке {patch_info['patch_name']}: {e}")
            embeddings_dict[patch_info['patch_name']] = ""
    
    return embeddings_dict

def process_patches_with_coordinates(data_path, map_path, sliced_path, patches_info, output_csv_path, city="austin"):
    """Создает CSV с координатами и эмбеддингами патчей"""
    print("Создание CSV с координатами патчей...")
    
    # Получаем границы города из исходных снимков
    map_left, map_bottom, map_right, map_top, crs = get_city_bounds(data_path, city)
    
    if None in [map_left, map_bottom, map_right, map_top]:
        print("Ошибка: не удалось получить границы города")
        return
    
    print(f"Границы города для расчетов:")
    print(f"  left={map_left:.1f}, bottom={map_bottom:.1f}")
    print(f"  right={map_right:.1f}, top={map_top:.1f}")
    
    print(f"Извлечение эмбеддингов для {len(patches_info)} патчей...")
    embeddings_dict = extract_embeddings_for_patches(patches_info)
    
    records = []
    
    for patch_info in patches_info:
        try:
            patch_name = patch_info['patch_name']
            i = patch_info['grid_i']
            j = patch_info['grid_j']
            patch_id = patch_info['patch_id']
            
            patch_size = 518
            num_patches = 58
            
            # Шаг в метрах (карта 30000 пикселей = 30000 метров)
            total_size = map_right - map_left  
            step = total_size / num_patches
            
            # X координаты (слева направо)
            patch_left = map_left + j * step
            patch_right = patch_left + step
            
            # Y координаты: инвертируем i!
            # Патч с i=0 должен быть вверху, i=57 внизу
            patch_bottom = map_bottom + (num_patches - 1 - i) * step
            patch_top = patch_bottom + step
            
            center_x = (patch_left + patch_right) / 2
            center_y = (patch_bottom + patch_top) / 2
            
            rel_x, rel_y = get_relative_coords(
                map_left, map_bottom, map_right, map_top,
                patch_left, patch_bottom, patch_right, patch_top
            )
            
            # Конвертируем в WGS84
            lat, lon = utm26910_to_wgs84(center_x, center_y, str(crs))
            
            embedding = embeddings_dict.get(patch_name, "")
            
            record = {
                "patch_id": patch_id,
                "patch_name": patch_name,
                "original_image": "austin_map",
                "patch_coords": f"{i}_{j}",
                "grid_i": i,
                "grid_j": j,
                "tile_path": str(patch_info['patch_path']),
                "lat": lat,
                "lon": lon,
                "patch_left": patch_left,
                "patch_bottom": patch_bottom,
                "patch_right": patch_right,
                "patch_top": patch_top,
                "rel_x": rel_x,
                "rel_y": rel_y,
                "split": "train",
                "embedding": embedding
            }
            
            records.append(record)
            
        except Exception as e:
            print(f"Ошибка при обработке {patch_info['patch_name']}: {e}")
    
    if records:
        df = pd.DataFrame(records)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Создан CSV: {output_csv_path} ({len(df)} записей)")
        
        # Проверка нескольких записей
        print("\nПримеры координат:")
        print(f"Патч 0 (0,0) - верхний левый: rel_x={df.iloc[0]['rel_x']:.4f}, rel_y={df.iloc[0]['rel_y']:.4f}")
        print(f"Патч 57 (0,57) - верхний правый: rel_x={df.iloc[57]['rel_x']:.4f}, rel_y={df.iloc[57]['rel_y']:.4f}")
        print(f"Патч 3306 (57,0) - нижний левый: rel_x={df.iloc[57*58]['rel_x']:.4f}, rel_y={df.iloc[57*58]['rel_y']:.4f}")

def create_train_data(city_name="austin"):
    """Создает полный датасет для одного города"""
    
    data_path = DATA_ROOT / "AerialImageDataset" / "train" / "images"
    
    maps_path = RESULTS_PATH / "maps"
    sliced_path = RESULTS_PATH / "sliced_images"
    sliced_city_data_path = RESULTS_PATH / "sliced_city_data"
    
    for path in [maps_path, sliced_path, sliced_city_data_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"Создание датасета для города {city_name}")
    
    print("1. Склеиваем карту города из тайлов")
    build_city_map(data_path, maps_path, city_name)
    build_city_compressed_map(data_path, maps_path, city_name)
    
    print("2. Нарезаем сшитую карту на патчи 518x518")
    num_patches, patches_info = slice_city_map(maps_path, sliced_path)
    
    print("3. Сшиваем патчи обратно в карту")
    stitch_patches_to_city(patches_info, maps_path, num_patches)
    
    print("4. Создаем CSV с координатами и эмбеддингами")
    csv_path = sliced_city_data_path / f"{city_name}_patches.csv"
    process_patches_with_coordinates(data_path, maps_path, sliced_path, patches_info, csv_path, city_name)
    
    print("Датасет создан")

if __name__ == "__main__":
    data_path = DATA_ROOT / "AerialImageDataset" / "train" / "images"
    map_left, map_bottom, map_right, map_top, crs = get_city_bounds(data_path, city="austin")

    lat1, lan1 = utm26910_to_wgs84(map_left, map_bottom) # Левый нижний
    lat2, lan2 = utm26910_to_wgs84(map_right, map_top) # Правый верхний
    lat3, lan3 = utm26910_to_wgs84((map_right+map_left)/2, (map_top + map_bottom)/2) # Центр

    print(f"Левый нижний угол широта/долгота: {lat1}/{lan1}")
    print(f"Правый верхний угол широта/долгота: {lat2}/{lan2}")
    print(f"Центр широта/долгота: {lat3}/{lan3}")
    