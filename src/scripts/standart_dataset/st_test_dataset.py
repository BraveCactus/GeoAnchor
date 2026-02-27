# create_val_data.py
import numpy as np
import tifffile as tiff
from pathlib import Path
import pandas as pd
import rasterio
from torchvision import transforms
from PIL import Image
import random
import json

from config import DATA_ROOT, RESULTS_PATH
from src.scripts.images_tools import utm26910_to_wgs84
from src.models.models import Dinov2EmbendingExtractor

def get_stitched_image_bounds_from_patches(city="austin"):
    """Определяет границы сшитого изображения по координатам нарезанных патчей"""    
    
    patches_csv_path = RESULTS_PATH / "sliced_city_data" / f"{city}_patches.csv"
    
    if not patches_csv_path.exists():
        print(f"Ошибка: CSV с патчами не найден: {patches_csv_path}")
        return None
    
    df = pd.read_csv(patches_csv_path)
    
    if len(df) == 0:
        print("Ошибка: CSV пустой")
        return None
    
    # Находим патчи с минимальными и максимальными координатами
    # Левый нижний патч (max i, min j) - потому что i=57 внизу, i=0 вверху
    # Ищем патч с максимальным grid_i (нижний ряд) и минимальным grid_j (левый столбец)
    bottom_left_patch = df[(df['grid_i'] == df['grid_i'].max()) & (df['grid_j'] == df['grid_j'].min())]
    if len(bottom_left_patch) == 0:
        bottom_left_patch = df[(df['grid_j'] == df['grid_j'].min())].iloc[-1]  # последний с минимальным j
    else:
        bottom_left_patch = bottom_left_patch.iloc[0]
    
    # Правый верхний патч (min i, max j) - i=0 вверху, j=max в правом столбце
    top_right_patch = df[(df['grid_i'] == df['grid_i'].min()) & (df['grid_j'] == df['grid_j'].max())]
    if len(top_right_patch) == 0:
        top_right_patch = df[(df['grid_j'] == df['grid_j'].max())].iloc[0]  # первый с максимальным j
    else:
        top_right_patch = top_right_patch.iloc[0]
    
    print("Определение границ сшитого изображения:")
    print(f"  Нижний левый патч: grid_i={bottom_left_patch['grid_i']}, grid_j={bottom_left_patch['grid_j']}")
    print(f"    patch_left={bottom_left_patch['patch_left']:.1f}, patch_bottom={bottom_left_patch['patch_bottom']:.1f}")
    
    print(f"  Верхний правый патч: grid_i={top_right_patch['grid_i']}, grid_j={top_right_patch['grid_j']}")
    print(f"    patch_right={top_right_patch['patch_right']:.1f}, patch_top={top_right_patch['patch_top']:.1f}")
    
    # Границы всего сшитого изображения
    map_left = float(bottom_left_patch['patch_left'])
    map_bottom = float(bottom_left_patch['patch_bottom'])
    map_right = float(top_right_patch['patch_right'])
    map_top = float(top_right_patch['patch_top'])
    
    print(f"\nГраницы сшитого изображения:")
    print(f"  left={map_left:.1f}, bottom={map_bottom:.1f}")
    print(f"  right={map_right:.1f}, top={map_top:.1f}")
    print(f"  Размер: {map_right-map_left:.1f}x{map_top-map_bottom:.1f} метров")
    
    # Получаем CRS из любого патча
    crs = "EPSG:26914"
    
    return {
        'left': map_left,
        'bottom': map_bottom,
        'right': map_right,
        'top': map_top,
        'crs': crs,
        'width_px': 30000,
        'height_px': 30000
    }

def extract_random_patches(image_path, output_path, map_bounds, num_patches=600, patch_size=518):
    """Вырезает случайные патчи 518x518 из сшитого изображения с правильными координатами"""
    
    # Загружаем изображение как обычный массив (без геопривязки)
    img_data = tiff.imread(image_path)
    
    # Проверяем размеры
    if len(img_data.shape) == 2:
        img_data = np.stack([img_data, img_data, img_data], axis=2)
    elif len(img_data.shape) == 3 and img_data.shape[2] > 3:
        img_data = img_data[:, :, :3]
    
    height, width = img_data.shape[:2]
    
    # Извлекаем границы
    map_left = map_bounds['left']
    map_bottom = map_bounds['bottom']
    map_right = map_bounds['right']
    map_top = map_bounds['top']
    crs = map_bounds['crs']
    
    print(f"Размер изображения: {width}x{height} пикселей")
    print(f"Масштаб: {(map_right - map_left) / width:.3f} м/пиксель по X, {(map_top - map_bottom) / height:.3f} м/пиксель по Y")
    
    patches_info = []
    
    for patch_num in range(num_patches):
        # Выбираем случайную позицию в пикселях
        max_y = height - patch_size
        max_x = width - patch_size
        
        y_start = random.randint(0, max_y)
        x_start = random.randint(0, max_x)
        y_end = y_start + patch_size
        x_end = x_start + patch_size
        
        # Вырезаем патч
        patch = img_data[y_start:y_end, x_start:x_end]
        
        patch_name = f"val_patch_{patch_num:03d}.tiff"
        patch_path = output_path / patch_name
        
        # Сохраняем патч
        tiff.imwrite(str(patch_path), patch)
        
        # Вычисляем координаты в метрах UTM
        pixel_to_meter_x = (map_right - map_left) / width
        pixel_to_meter_y = (map_top - map_bottom) / height
        
        # X координата (запад-восток)
        patch_left = map_left + x_start * pixel_to_meter_x
        patch_right = map_left + x_end * pixel_to_meter_x
        
        # Y координата: в растровых данных Y идет сверху вниз (0 вверху), 
        # но в UTM координаты идут снизу вверх (0 внизу)
        # Поэтому: map_top - y * pixel_to_meter_y
        patch_top = map_top - y_start * pixel_to_meter_y
        patch_bottom = map_top - y_end * pixel_to_meter_y
        
        # Убедимся, что bottom < top
        if patch_bottom > patch_top:
            patch_bottom, patch_top = patch_top, patch_bottom
        
        patches_info.append({
            'patch_id': patch_num,
            'patch_name': patch_name,
            'original_image': Path(image_path).stem,
            'patch_path': patch_path,
            'pixel_y_start': y_start,
            'pixel_x_start': x_start,
            'patch_left': patch_left,
            'patch_bottom': patch_bottom,
            'patch_right': patch_right,
            'patch_top': patch_top,
            'crs': crs
        })
        
        if patch_num % 100 == 0:
            print(f"  Создано {patch_num}/{num_patches} патчей")
    
    print(f"  Всего создано {len(patches_info)} патчей")
    return patches_info

def get_relative_coords(map_left, map_bottom, map_right, map_top, 
                       patch_left, patch_bottom, patch_right, patch_top):
    """Относительные координаты центра патча (0-1)"""
    # Центр патча
    center_x = (patch_left + patch_right) / 2
    center_y = (patch_bottom + patch_top) / 2
    
    # Относительные координаты
    width = map_right - map_left
    height = map_top - map_bottom
    
    rel_x = (center_x - map_left) / width
    rel_y = (center_y - map_bottom) / height
    
    # Проверка диапазона
    rel_x = max(0.0, min(1.0, rel_x))
    rel_y = max(0.0, min(1.0, rel_y))
    
    return rel_x, rel_y

def extract_embeddings_for_patches(patches_info):
    """Извлекает эмбеддинги для патчей"""
    
    print("Извлечение эмбеддингов DINOv2...")
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
            
            # Конвертируем в 3 канала если нужно
            if len(img_data.shape) == 2:
                img_data = np.stack([img_data, img_data, img_data], axis=2)
            elif img_data.shape[2] > 3:
                img_data = img_data[:, :, :3]
            
            pil_img = Image.fromarray(img_data)
            
            # Ресайз для DINOv2
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

def create_test_data(num_patches=600):
    """Создает валидационный датасет из austin_stitched_from_patches.tiff"""
    
    print("Создание валидационного датасета")
    print("=" * 50)
    
    # Используем сшитое изображение
    image_path = RESULTS_PATH / "maps" / "austin_stitched_from_patches.tiff"
    
    if not image_path.exists():
        print(f"Ошибка: файл не найден: {image_path}")
        print("Сначала запустите create_dataset.py для создания сшитого изображения")
        return
    
    # Создаем директории
    val_path = RESULTS_PATH / "validation"
    val_images_path = val_path / "images"
    val_data_path = val_path / "data"
    
    for path in [val_path, val_images_path, val_data_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # 1. Определяем границы сшитого изображения по патчам
    map_bounds = get_stitched_image_bounds_from_patches(city="austin")
    
    if map_bounds is None:
        print("Не удалось определить границы. Используем приблизительные значения...")
        map_bounds = {
            'left': 620000,
            'bottom': 3340000,
            'right': 650000,
            'top': 3370000,
            'crs': "EPSG:26914",
            'width_px': 30000,
            'height_px': 30000
        }
    
    # 2. Вырезаем случайные патчи
    print(f"Вырезание {num_patches} патчей из {image_path}")
    patches_info = extract_random_patches(image_path, val_images_path, map_bounds, num_patches)
    
    if not patches_info:
        print("Ошибка: не удалось вырезать патчи")
        return
    
    # 3. Извлекаем эмбеддинги
    embeddings_dict = extract_embeddings_for_patches(patches_info)
    
    # 4. Создаем CSV с координатами
    records = []
    
    map_left = map_bounds['left']
    map_bottom = map_bounds['bottom']
    map_right = map_bounds['right']
    map_top = map_bounds['top']
    
    for patch_info in patches_info:
        patch_name = patch_info['patch_name']
        
        # Координаты центра патча
        center_x = (patch_info['patch_left'] + patch_info['patch_right']) / 2
        center_y = (patch_info['patch_bottom'] + patch_info['patch_top']) / 2
        
        # Относительные координаты
        rel_x, rel_y = get_relative_coords(
            map_left, map_bottom, map_right, map_top,
            patch_info['patch_left'], patch_info['patch_bottom'],
            patch_info['patch_right'], patch_info['patch_top']
        )
        
        # Конвертируем в WGS84
        try:
            lat, lon = utm26910_to_wgs84(center_x, center_y, str(patch_info['crs']))
        except Exception as e:
            print(f"Ошибка преобразования координат для {patch_name}: {e}")
            lat, lon = 0.0, 0.0
        
        embedding = embeddings_dict.get(patch_name, "")
        
        record = {
            "patch_id": patch_info['patch_id'],
            "patch_name": patch_name,
            "original_image": patch_info['original_image'],
            "patch_coords": f"{patch_info['pixel_y_start']}_{patch_info['pixel_x_start']}",
            "tile_path": str(patch_info['patch_path']),
            "lat": lat,
            "lon": lon,
            "center_x_utm": center_x,
            "center_y_utm": center_y,
            "patch_left": patch_info['patch_left'],
            "patch_bottom": patch_info['patch_bottom'],
            "patch_right": patch_info['patch_right'],
            "patch_top": patch_info['patch_top'],
            "rel_x": rel_x,
            "rel_y": rel_y,
            "split": "val",
            "embedding": embedding
        }
        
        records.append(record)
    
    # 5. Сохраняем в CSV
    df = pd.DataFrame(records)
    csv_path = val_data_path / "validation_patches.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nВалидационный датасет создан успешно!")
    print(f"Создано патчей: {len(df)}")
    print(f"CSV файл: {csv_path}")
    print(f"Изображения: {val_images_path}")
    
    # 6. Выводим статистику
    print("\nСтатистика координат:")
    print("-" * 40)
    print(f"Диапазон широт: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
    print(f"Диапазон долгот: [{df['lon'].min():.6f}, {df['lon'].max():.6f}]")
    print(f"Диапазон rel_x: [{df['rel_x'].min():.4f}, {df['rel_x'].max():.4f}]")
    print(f"Диапазон rel_y: [{df['rel_y'].min():.4f}, {df['rel_y'].max():.4f}]")
    
    return csv_path

if __name__ == "__main__":
    # Создаем валидационный датасет (600 изображений)
    create_val_dataset(num_patches=600)