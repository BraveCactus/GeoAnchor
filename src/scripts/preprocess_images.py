import os
import numpy as np
import pandas as pd
import rasterio
import tifffile as tiff
from PIL import Image
from torchvision import transforms

from config import PREPROCESSES_IMAGES_PATH
from src.scripts.images_tools import read_tiff_metadata, utm26910_to_wgs84
from src.models.models import Dinov2EmbendingExtractor

def show_image_data(img):
    """Показывает данные изображения"""

    print("=" * 50)
    print("ИНФОРМАЦИЯ О ЗАГРУЖЕННОМ ИЗОБРАЖЕНИИ:")
    print("=" * 50)
    print(f"1. Тип объекта: {type(img)}")
    print(f"2. Тип изображения (бит на пиксель): {img.dtype}")
    print(f"3. Форма массива (shape): {img.shape}")
    print(f"   - Высота: {img.shape[0]} пикселей")
    print(f"   - Ширина: {img.shape[1]} пикселей")

def preprocess_image(img_path, target_size=518):
    """Функция переводит изображение в формат, пригодный для модели DINOv2 (518×518, 3 канала, нормировка)"""
    try:
        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")          
            return None, None
        
        img_data = tiff.imread(img_path)

        # Обработка каналов
        if len(img_data.shape) == 2:  
            img_data = np.stack([img_data, img_data, img_data], axis=2)
        elif len(img_data.shape) == 3:
            if img_data.shape[2] > 3:
                img_data = img_data[:, :, :3]  
            elif img_data.shape[2] == 1: 
                img_data = np.stack([img_data, img_data, img_data], axis=2)
            elif img_data.shape[2] == 3:
                pass 
            else:
                print(f"Неожиданное кол-во каналов: {img_data.shape[2]}")
                return None, None
        else:
            print(f"Неожиданная размерность: {img_data.shape}")
            return None, None
        
        # Конвертация в uint8
        if img_data.dtype == np.uint16:            
            img_data = (img_data / 65535.0 * 255).astype(np.uint8)
        elif img_data.dtype != np.uint8:            
            img_data = img_data.astype(np.uint8)

        # Меняем размер изображения на нужный
        pil_img = Image.fromarray(img_data)

        if (img_data.shape[0] != target_size or img_data.shape[1] != target_size):            
            resized_pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            resized_img_data = np.array(resized_pil_img)
        else:
            resized_pil_img = pil_img  
            resized_img_data = img_data 

        # Нормализуем изображение
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        img_tensor = transform(resized_pil_img)

        # Сохраняем результат         
        os.makedirs(PREPROCESSES_IMAGES_PATH, exist_ok=True)

        original_filename = os.path.splitext(os.path.basename(img_path))[0]
        new_filename = f"{original_filename}_processed.tiff"
        save_path = os.path.join(PREPROCESSES_IMAGES_PATH, new_filename)
        
        tiff.imwrite(save_path, resized_img_data)
        print(f"Обработанное изображение {new_filename} сохранено в {PREPROCESSES_IMAGES_PATH}")      

        return img_tensor, resized_img_data

    except Exception as e:
        print(f"Произошла ошибка при загрузке изображения: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_embedding_from_image(img_path, extractor=None):
    """Извлекаем embedding из изображения"""
    
    img_tensor, _ = preprocess_image(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    if img_tensor is None:
        print(f"Не удалось обработать {img_path}")
        return None
    
    if extractor is None:
        extractor = Dinov2EmbendingExtractor()        

    embedding_array = extractor.extract_embedding(img_tensor)

    print(f"Извлечен эмбеддинг для изображения {os.path.basename(img_path)}")      

    return img_name, embedding_array

def extract_embedding_from_city(city_path, output_path, city="austin"):
    """Извлекаем embedding из всех изображений города"""
    output_path.mkdir(exist_ok=True, parents=True)
    city_imgs = [img_file for img_file in os.listdir(city_path) if img_file.startswith(city)]

    print(f"Извлекаем embedding из снимков {city}")

    records = []
    for city_img in city_imgs:
        img_file_path = city_path / city_img
        embedding_result = extract_embedding_from_image(img_file_path)
        
        if embedding_result is not None:
            img_name, embedding_array = embedding_result
            records.append({
                "image_name": img_name,
                "embedding": ';'.join(map(str, embedding_array))
            })

    df_embending = pd.DataFrame(records)
    save_path = output_path / f"{city}_embending.csv"
    df_embending.to_csv(save_path, index=False)
    print(f"Извлечение embedding из снимков {city} произошло успешно")
    return df_embending

def slice_image(img_path, output_path, new_side=518):
    """Нарезает снимок на более мелкие"""
    output_path.mkdir(parents=True, exist_ok=True)
    img = tiff.imread(img_path)  
    img_name = os.path.basename(img_path)

    if (img.shape[2] != 3 or 
        img.shape[0] != 5000 or
        img.shape[1] != 5000 or
        img.dtype != np.uint8):
        print(f"Изображение {img_name} не подходит для нарезки из-за некорректного формата")
        return []   
    
    if (img.shape[0] // new_side == 0 or img.shape[1] // new_side == 0):
        print(f"Размер изображения {img_name} слишком мал для нарезки на куски {new_side}x{new_side}")
        return []
    
    pil = Image.fromarray(img)
    
    new_size = (img.shape[0] // new_side) * new_side 
    pil = pil.resize((new_size, new_size), Image.Resampling.LANCZOS)

    img_small = np.array(pil)

    patches = []

    for i in range(0, new_size, new_side):
        for j in range(0, new_size, new_side):
            patch = img_small[i:i+new_side, j:j+new_side]
            patch_name = f"{img_name[:-4]}_patch_{i//new_side}_{j//new_side}.tiff"
            tiff.imwrite(os.path.join(output_path, patch_name), patch)
            patches.append(patch)

    return patches

def slice_one_city(city_path, output_path, city="austin"):
    """Нарезает снимки одного города на более мелкие"""
    city_imgs = [img_file for img_file in os.listdir(city_path) if img_file.startswith(city)]
    for city_img in city_imgs:
        img_file_path = city_path / city_img
        slice_image(img_file_path, output_path)

    print(f"Нарезка города {city} завершена")

def full_process_image(img_path, index, split="train"):
    """Полная обработка изображения: embedding + метаданные"""    
    embedding_result = extract_embedding_from_image(img_path)
    
    if embedding_result is None:
        print(f"Не удалось извлечь embedding для {img_path}")
        return None    
    
    img_name, embedding_array = embedding_result    
    
    embedding_str = ';'.join(map(str, embedding_array))
    
    try:        
        metadata = read_tiff_metadata(img_path)
        left, bottom, right, top = metadata["bounds"]
        center_x_NAD83 = (left + right) / 2
        center_y_NAD83 = (bottom + top) / 2        
        
        city_zone = str(metadata['crs'])
        
        lat, lon = utm26910_to_wgs84(center_x_NAD83, center_y_NAD83, city_zone)
        
        record = {
            "tile_name": f"{img_name}_{index}",
            "image_name": img_name,
            "tile_path": str(img_path),
            "lat": lat,
            "lon": lon,
            "split": split,
            "embedding": embedding_str
        }
        
        return record
        
    except Exception as e:
        print(f"Ошибка при обработке метаданных {img_path}: {e}")
        return None

def process_city(city_path, output_path, city="austin", split="train"):
    """Обрабатывает все изображения города и создает CSV с данными"""
    output_path.mkdir(exist_ok=True, parents=True)
    city_imgs = [img_file for img_file in os.listdir(city_path) if img_file.startswith(city)]
    records = []
    
    print(f"Обрабатываем город {city} ({len(city_imgs)} изображений)...")
    
    for idx, img_file in enumerate(city_imgs):    
        img_file_path = city_path / img_file
        img_data = full_process_image(img_file_path, idx, split)
        if img_data is not None:
            records.append(img_data)
        else:
            print(f"Пропущено изображение: {img_file}")

    if records:
        df = pd.DataFrame(records)
        save_path = output_path / f"{city}_data.csv"
        df.to_csv(save_path, index=False)
        print(f"Обработано {len(records)} изображений города {city}")
        print(f"Данные сохранены в {save_path}")
        return df
    else:
        print(f"Не удалось обработать ни одного изображения города {city}")
        return None