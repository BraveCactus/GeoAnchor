import os
import io
import sys
import numpy as np
import tifffile as tiff
from PIL import Image
from torchvision import transforms

from config import DATA_ROOT, PREPROCESSES_IMAGES_ROOT

def show_image_data(img_data):
    """Показывает данные изображения"""

    print("=" * 50)
    print("ИНФОРМАЦИЯ О ЗАГРУЖЕННОМ ИЗОБРАЖЕНИИ:")
    print("=" * 50)
    print(f"1. Тип объекта: {type(img_data)}")
    print(f"2. Тип изображения (бит на пиксель): {img_data.dtype}")
    print(f"3. Форма массива (shape): {img_data.shape}")
    print(f"   - Высота: {img_data.shape[0]} пикселей")
    print(f"   - Ширина: {img_data.shape[1]} пикселей")

def load_and_preprocess_tiff(tiff_img_path, target_size=518):
    """Функция переводит изображение в формат, пригодный для модели DINOv2"""
    try:

        if not os.path.exists(tiff_img_path):
            print(f"Файл не найден: {tiff_img_path}")          
            return None, None
        img_data = tiff.imread(tiff_img_path)

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
            img_data = (img_data / 65535.0 * 255).astype(np.uint8)  # ИСПРАВЛЕНО
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
            )
        ])

        img_tensor = transform(resized_pill_img)
        # print(f"Тензор после нормализации: shape={img_tensor.shape}, "
        #       f"диапазон=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

        # Сохраняем результат         
        os.makedirs(PREPROCESSES_IMAGES_ROOT, exist_ok=True)

        original_filename = (os.path.basename(tiff_img_path)).split(".")[0]
        new_filename = f"{original_filename}_processed.tiff"

        save_path = os.path.join(PREPROCESSES_IMAGES_ROOT, new_filename)
        
        tiff.imwrite(save_path, resized_img_data)
        print(f"Обработанное изображение {new_filename} сохранено в {PREPROCESSES_IMAGES_ROOT}")      

        return img_tensor, resized_img_data

    except Exception as e:
        print(f"Произошла ошибка при загрузке изображения: {e}")
