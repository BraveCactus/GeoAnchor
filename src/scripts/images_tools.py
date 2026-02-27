import pyproj
import rasterio
import os
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from config import SEQUENCE

# def utm_to_wgs84(easting, northing, crs):
#     transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
#     lon, lat = transformer.transform(easting, northing)
#     return lat, lon

def view_geotiff(filepath):
    """Параметры изображения формата TIFF"""

    with rasterio.open(filepath) as src:
        left, bottom, right, top = src.bounds

        lan1, lat1 = utm26910_to_wgs84(left, bottom)
        lan2, lat2 = utm26910_to_wgs84(right, top)

        center_x = (left + right) / 2
        center_y = (bottom + top) / 2

        lan, lat = utm26910_to_wgs84(center_x, center_y)
    
    with rasterio.open(filepath) as src:
        print("Метаданные GeoTIFF:")
        print(f"Размер: {src.width}×{src.height}")
        print(f"Количество каналов: {src.count}")
        print(f"Тип данных: {src.dtypes}")
        print(f"CRS (система координат): {src.crs}")
        print(f"Границы: {src.bounds}")
        print(f"Левый нижний угол: {lan1}/{lat1}")
        print(f"Правый верхний угол: {lan2}/{lat2}")
        print(f"Широта/Долгота: {lan}/{lat}")
        print(f"Разрешение: {src.res} метров/пиксель")

def show_tiff_image(filepath):    
    """Показывает изображение"""
    
    image = Image.open(filepath)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"TIFF: {image.size[0]}×{image.size[1]}, {image.mode}")
    plt.axis('off')    
    plt.show()

def read_tiff_metadata(img_path):
    """Получает метаданные об изображении формата tiff"""
    with rasterio.open(img_path) as src:
        left, bottom, right, top = src.bounds
    
        return {
            'crs': src.crs,
            'bounds': src.bounds,
            'transform': src.transform,
            'width': src.width,
            'height': src.height
        }
    
def utm26910_to_wgs84(easting, northing, city_zone="EPSG:26914"):
    """Перевод американской системы координат в нормальную (широта, долгота)"""
    transformer = pyproj.Transformer.from_crs(
        city_zone,  
        "EPSG:4326",
        always_xy=True
    )

    lon, lat = transformer.transform(easting, northing)
    
    return lat, lon

def get_city_data(data_path, output_path, city="austin"):
    """Получаем данные о городе и сохраняем в csv-файл"""
    output_path.mkdir(parents=True, exist_ok=True)

    city_files = list(data_path.glob(f"{city}*.tif"))
    data_records = []    

    print(f"Получаем данные о городе {city}")
    for img_path in city_files:
        metadata = read_tiff_metadata(img_path)
        left, bottom, right, top = metadata["bounds"]

        center_x_NAD83 = (left + right) / 2
        center_y_NAD83 = (bottom + top) / 2

        city_zone = str(metadata['crs'])

        lat, lon = utm26910_to_wgs84(center_x_NAD83, center_y_NAD83, city_zone)
        
        img_data_dict = {
            "image_name": img_path.name,
            "city": city,
            'crs': city_zone,
            'left': left,
            'bottom': bottom, 
            'right': right,
            'top': top,
            'center_x_NAD83': center_x_NAD83,
            'center_y_NAD83': center_y_NAD83,
            'center_latitude': lat,
            'center_longitude': lon,
            'width': metadata['width'],
            'height': metadata['height']
        }

        data_records.append(img_data_dict)

    df = pd.DataFrame(data_records)

    save_path = output_path / f"{city}_data.csv"
    df.to_csv(save_path, index=False)
    print(f"Создан {city}_data.csv: {len(df)} записей")    

def build_city_map(data_path, output_path, city="austin"):
    """Склеивает город из 36 больших снимков и сохраняет готовое изображение"""
    output_path.mkdir(parents=True, exist_ok=True)

    tile_size = 5000
    grid_size = 6
    final_size = tile_size * grid_size

    stitched_image = np.zeros((final_size, final_size, 3), dtype=np.uint8)  
    
    for idx, tile_number in enumerate(SEQUENCE):        
        row = idx // grid_size  
        col = idx % grid_size 

        filename = f"{city}{tile_number}.tif"
        file_path = data_path / filename

        y_start = row * tile_size
        y_end = y_start + tile_size
        x_start = col * tile_size
        x_end = x_start + tile_size

        try:
            tile_data = tiff.imread(file_path)
            if (tile_data.shape[2] != 3 or 
                tile_data.shape[0] != 5000 or
                tile_data.shape[1] != 5000 or
                tile_data.dtype != np.uint8):
                print(f"Изображение номер {tile_number} не подходит для построение карты города {city}")
                return
            
            stitched_image[y_start:y_end, x_start:x_end] = tile_data
        except Exception as e:
            print(f"Ошибка сшивки изображения {filename}: {e}")

    output_filename = f"{city}_map.tiff"
    output_path = output_path / output_filename

    tiff.imwrite(str(output_path), stitched_image,  compression='lzw')
    print(f"Карта города {city} сшита")

def build_city_compressed_map(data_path, output_path, city="austin", scale=0.25):
    """
    Склеивает город в уменьшенном размере    
    """
    output_path.mkdir(parents=True, exist_ok=True)    
    
    small_tile = int(5000 * scale)
    small_size = int(30000 * scale)    
    
    small_image = np.zeros((small_size, small_size, 3), dtype=np.uint8)
    
    print(f"Сборка {city} в масштабе {scale*100}%...")
    
    for idx, tile_number in enumerate(SEQUENCE):        
        row = idx // 6  
        col = idx % 6 

        filename = f"{city}{tile_number}.tif"
        file_path = data_path / filename

        try:
            tile_data = tiff.imread(str(file_path))           
            
            if (tile_data.shape[2] != 3 or  # Исправлено с 2 на 3 для RGB
                tile_data.shape[0] != 5000 or
                tile_data.shape[1] != 5000 or
                tile_data.dtype != np.uint8):
                print(f"Изображение номер {tile_number} не подходит для построение карты города {city}")
                return            
            
            pil_tile = Image.fromarray(tile_data)
            pil_tile = pil_tile.resize((small_tile, small_tile), Image.Resampling.LANCZOS)
            small_tile_rgb = np.array(pil_tile)            
            
            y_start = row * small_tile
            y_end = y_start + small_tile
            x_start = col * small_tile
            x_end = x_start + small_tile            
            
            small_image[y_start:y_end, x_start:x_end] = small_tile_rgb
            
        except Exception as e:
            print(f"Ошибка сшивки изображения {filename}: {e}")    
    
    save_path = output_path / f"{city}_map_{small_size}x{small_size}.tiff"     
    
    tiff.imwrite(
        str(save_path), 
        small_image,
        compression='lzw'
    ) 

    print(f"Сшивка карты {city} завершена!")  

def analyze_city_coordinates(csv_path):
    """Анализирует координаты снимков города"""
    df = pd.read_csv(csv_path)
    
    print("Анализ координат города:")
    print("-" * 40)    
    
    min_left = df['left'].min()
    min_bottom = df['bottom'].min()
    max_right = df['right'].max()
    max_top = df['top'].max()
    
    print(f"Левая граница: {min_left:.1f}")
    print(f"Правая граница: {max_right:.1f}")
    print(f"Нижняя граница: {min_bottom:.1f}")
    print(f"Верхняя граница: {max_top:.1f}")    
    
    unique_left = sorted(df['left'].unique())
    unique_bottom = sorted(df['bottom'].unique())
    
    if len(unique_left) > 1:
        step_x = unique_left[1] - unique_left[0]
        print(f"Шаг по X: {step_x:.1f} м")
    
    if len(unique_bottom) > 1:
        step_y = unique_bottom[1] - unique_bottom[0]
        print(f"Шаг по Y: {step_y:.1f} м")
    
    # Определяем сетку
    grid_width = len(unique_left)
    grid_height = len(unique_bottom)
    print(f"Сетка снимков: {grid_width}×{grid_height}")
    
    return {
        'min_left': min_left,
        'max_right': max_right,
        'min_bottom': min_bottom,
        'max_top': max_top,
        'grid_size': (grid_width, grid_height)
    }

def extract_center(image_path, output_path=None):    

    img = tiff.imread(image_path)
    target_size = 4662

    pil_img = Image.fromarray(img)

    resized_pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    h, w = resized_pil_img.shape[0], resized_pil_img.shape[1]
    h_center, w_center = h // 2, w // 2
    
    patch = resized_pil_img[h_center-259:h_center+259, w_center-259:w_center+259]
    
    if output_path:
        tiff.imwrite(output_path, patch)
    
    return patch

def extract_center_images(image_path, output_folder):   
    
    img = tiff.imread(image_path)    
    
    with rasterio.open(image_path) as src:
        transform = src.transform  
        crs = src.crs  
        left, bottom, right, top = src.bound

  
    center_y, center_x = (top + bottom) / 2 , (left + right) / 2
    
    patch_size = 518
    half_patch = patch_size // 2    
    
    # grid_offsets = [
    #     (-518, -518), (0, -518), (518, -518),
    #     (-518, 0),    (0, 0),    (518, 0),
    #     (-518, 518),  (0, 518),  (518, 518)
    # ]

    grid_offsets = [
        (0, 0),                
        (0, -518), (518, 0), (0, 518), (-518, 0),  
        (-518, -518), (0, -1036), (518, -518),     
        (1036, 0), (518, 518), (0, 1036), (-518, 518), (-1036, 0)
    ]
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    patches_data = []    
    
    for idx, (offset_y, offset_x) in enumerate(grid_offsets):        
       
        start_y = center_y + offset_y - half_patch
        end_y = start_y + patch_size
        start_x = center_x + offset_x - half_patch
        end_x = start_x + patch_size        
        
        patch = img[start_y:end_y, start_x:end_x]        
       
        patch_name = f"patch_{idx}.tiff"
        patch_path = output_path / patch_name
        tiff.imwrite(patch_path, patch)        
        
        left, top = transform * (start_x, start_y)     
        right, bottom = transform * (end_x, end_y)              
        
        center_geo_x = (left + right) / 2
        center_geo_y = (top + bottom) / 2      

        lat, lon = utm26910_to_wgs84(center_geo_x, center_geo_y)  
        
        patches_data.append({
            'patch_id': idx,
            'patch_name': patch_name,
            'file_path': str(patch_path),
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom,
            'center_x': center_geo_x,
            'center_y': center_geo_y,
            'lat': lat,
            'lon': lon
        })    
    
    df = pd.DataFrame(patches_data)    
    
    csv_path = output_path / "patches_coordinates.csv"
    df.to_csv(csv_path, index=False)
    
    return df

# def read_coordinates(csv_path):
#     """Читает координаты из CSV в указанном порядке"""
#     coordinates = []
    
#     with open(csv_path, 'r') as f:
#         reader = csv.DictReader(f)
        
#         # Читаем все данные в словарь
#         data = {}
#         for row in reader:
#             name = row['patch_name']
#             lat = float(row['lat'])
#             lon = float(row['lon'])
#             data[name] = (lat, lon)
    
#     # Определяем порядок считывания
#     base_names = ['austin6', 'austin12', 'austin18', 'austin24', 'austin30', 'austin36']
    
#     for base in base_names:
#         # Для каждого базового имени берем 9 снимков
#         for i in range(9):  # 0-8
#             patch_name = f"{base}_patch_0_{i}"
#             if patch_name in data:
#                 lat, lon = data[patch_name]
#                 coordinates.append((patch_name, lat, lon))
    
#     return coordinates


