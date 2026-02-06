import os
import numpy as np
import pandas as pd
import rasterio
import tifffile as tiff
from PIL import Image
from torchvision import transforms

from config import DATA_ROOT, PREPROCESSES_IMAGES_PATH, SEQUENCE
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

def load_and_preprocess_tiff(img_path, target_size=518):
    """Функция переводит изображение в формат, пригодный для модели DINOv2 (518×518, 3 канала, нормировка)"""
    try:
        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")          
            return None, None
        img_data = tiff.imread(img_path)

        # Оставлям 3 канала
        if img_data.shape[2] > 3:
            img_data = img_data[:, :, :3]
        elif img_data.shape[2] == 3:
            pass
        else:
            print("Неожиданное кол-во каналов")
            return None, None
        
        # Делаем так, чтобы на каждый пиксель приходилось по 8 бит        
        if img_data.dtype == np.uint16:            
            img_data = (img_data / 65535.0 * 255).astype(np.uint8)
        elif img_data.dtype == np.uint8:            
            pass
        else:            
            img_data = img_data.astype(np.uint8)

        # Меняем размер изображения на нужный
        pill_img = Image.fromarray(img_data)

        if (img_data.shape[0] != target_size or img_data.shape[1] != target_size):            
            resized_pill_img = pill_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            resized_img_data = np.array(resized_pill_img)
            # print(f"Новый размер: {resized_img_data.shape}")
        else:
            resized_pill_img = pill_img  
            resized_img_data = img_data 

        # Нормализуем изображение
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) # норм_знач = (ориг_знач - mean) / std
        ])

        img_tensor = transform(resized_pill_img)
        # print(f"Тензор после нормализации: shape={img_tensor.shape}, "
        #       f"диапазон=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

        # Сохраняем результат         
        os.makedirs(PREPROCESSES_IMAGES_PATH, exist_ok=True)

        original_filename = (os.path.basename(img_path)).split(".")[0]
        new_filename = f"{original_filename}_processed.tiff"

        save_path = os.path.join(PREPROCESSES_IMAGES_PATH, new_filename)
        
        tiff.imwrite(save_path, resized_img_data)
        print(f"Обработанное изображение {new_filename} сохранено в {PREPROCESSES_IMAGES_PATH}")      

        return img_tensor, resized_img_data

    except Exception as e:
        print(f"Произошла ошибка при загрузке изображения: {e}")

def extract_embedding_from_tiff(img_path, extractor=None):
    """Извлекаем embedding из tiff изображения"""
    img_tensor, _ = load_and_preprocess_tiff(img_path)

    if img_tensor is None:
        print(f"Не удалось обработать {img_path}")
        return None
    
    if extractor == None:
        extractor = Dinov2EmbendingExtractor()

    embedding = extractor.extract_embedding(img_tensor)

    print(f"Извлечен эмбеддинг для изображения {os.path.basename(img_path)}")    

    return embedding

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
    pil = pil.resize((new_size, new_size))

    img_small = np.array(pil)

    patches = list()

    for i in range(0, new_size, new_side):
        for j in range(0, new_size, new_side):
            patch = img_small[i:i+new_side, j:j+new_side]
            tiff.imwrite(f"{output_path}/{img_name[:-4]}_patch_{i//new_side}_{j//new_side}.tiff", patch)
            patches.append(patch)

    return patches


