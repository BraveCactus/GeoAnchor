import os
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from torchvision import transforms

from src.scripts.images_tools import read_tiff_metadata, utm26910_to_wgs84
from src.models.models import Dinov2EmbendingExtractor

def process_sliced_image(img_path, extractor=None):
    """
    Обрабатывает нарезанный патч: извлекает embedding
    
    Args:
        img_path: Path к патчу
        extractor: экстрактор эмбеддингов
    
    Returns:
        tuple: (img_name, embedding_array)
    """
    try:
        # Загружаем и предобрабатываем изображение
        img_tensor = preprocess_sliced_tiff(img_path)
        if img_tensor is None:
            return None
        
        # Извлекаем embedding
        if extractor is None:
            extractor = Dinov2EmbendingExtractor()
        
        embedding_array = extractor.extract_embedding(img_tensor)
        
        # Получаем имя файла
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        return img_name, embedding_array
        
    except Exception as e:
        print(f"Ошибка при обработке патча {img_path}: {e}")
        return None

def preprocess_sliced_tiff(img_path, target_size=518):
    """
    Предобработка нарезанного TIFF файла для DINOv2
    """
    try:
        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")          
            return None
        
        img_data = tiff.imread(img_path)
        
        # Обработка каналов
        if len(img_data.shape) == 2:  # черно-белое
            img_data = np.stack([img_data, img_data, img_data], axis=2)
        elif len(img_data.shape) == 3:
            if img_data.shape[2] > 3:
                img_data = img_data[:, :, :3]
            elif img_data.shape[2] == 1:
                img_data = np.stack([img_data, img_data, img_data], axis=2)
        else:
            return None
        
        # Конвертация в uint8
        if img_data.dtype == np.uint16:            
            img_data = (img_data / 65535.0 * 255).astype(np.uint8)
        elif img_data.dtype != np.uint8:            
            img_data = img_data.astype(np.uint8)
        
        # Ресайз если нужно
        pil_img = Image.fromarray(img_data)
        if img_data.shape[0] != target_size or img_data.shape[1] != target_size:
            pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Нормализация
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(pil_img)
        return img_tensor
        
    except Exception as e:
        print(f"Ошибка при предобработке {img_path}: {e}")
        return None

def process_sliced_with_coordinates(sliced_images_path, original_csv_path, output_path, city="austin", split="train"):
    """
    Обрабатывает нарезанные изображения с вычислением координат
    """
    try:        
        original_df = pd.read_csv(original_csv_path)        
        
        print(f"Колонки в оригинальном CSV: {list(original_df.columns)}")        
        
        image_name_col = None
        for col in ['image name', 'image_name', 'name']:
            if col in original_df.columns:
                image_name_col = col
                break
        
        if image_name_col is None:
            print(f"Не найдена колонка с именами изображений. Доступные колонки: {list(original_df.columns)}")
            return None        
        
        original_coords = {}
        for _, row in original_df.iterrows():
            img_name = row[image_name_col].replace('.tif', '').replace('.tiff', '')
            original_coords[img_name] = {
                'left': row['left'],
                'bottom': row['bottom'],
                'right': row['right'],
                'top': row['top'],
                'crs': row['crs'] if 'crs' in row else 'EPSG:26910' 
            }
        
        print(f"Загружено {len(original_coords)} оригинальных изображений")        
        
        pattern = f"{city}*_patch_*.tiff"
        patch_files = list(sliced_images_path.glob(pattern))
        
        if not patch_files:
            print(f"Не найдено патчей для города {city} в {sliced_images_path}")
            return None
        
        print(f"Найдено {len(patch_files)} патчей для города {city}")
        
        records = []
        extractor = Dinov2EmbendingExtractor()
        
        for idx, patch_file in enumerate(patch_files):
            if idx % 10 == 0:
                print(f"Обработка патча {idx+1}/{len(patch_files)}...")
            
            # Извлекаем embedding
            embedding_result = process_sliced_image(patch_file, extractor)
            if embedding_result is None:
                continue
                
            patch_name, embedding_array = embedding_result
            
            # Преобразуем embedding в строку
            embedding_str = ';'.join(map(str, embedding_array))
            
            # Получаем имя оригинального изображения и координаты патча
            try:
                # patch_name = "austin1_patch_0_0"
                original_img = patch_name.split('_patch_')[0]
                patch_coords = patch_name.split('_patch_')[1].split('_')  # ["0", "0"]
                
                if original_img in original_coords and len(patch_coords) == 2:
                    i, j = int(patch_coords[0]), int(patch_coords[1])
                    orig = original_coords[original_img]
                    
                    # Вычисляем координаты патча                    
                    num_patches = 5000 // 518 
                    
                    patch_width = (orig['right'] - orig['left']) / num_patches
                    patch_height = (orig['top'] - orig['bottom']) / num_patches
                    
                    patch_left = orig['left'] + j * patch_width
                    patch_right = patch_left + patch_width
                    patch_bottom = orig['bottom'] + i * patch_height
                    patch_top = patch_bottom + patch_height
                    
                    # Центр патча
                    center_x = (patch_left + patch_right) / 2
                    center_y = (patch_bottom + patch_top) / 2
                    
                    # Конвертируем в WGS84
                    lat, lon = utm26910_to_wgs84(center_x, center_y, orig['crs'])
                    
                    record = {
                        "patch_name": patch_name,
                        "original_image": original_img,
                        "patch_coords": f"{i}_{j}",
                        "tile_path": str(patch_file),
                        "lat": lat,
                        "lon": lon,
                        "patch_left": patch_left,
                        "patch_bottom": patch_bottom,
                        "patch_right": patch_right,
                        "patch_top": patch_top,
                        "split": split,
                        "embedding": embedding_str
                    }
                    
                    records.append(record)
                else:
                    print(f"Не найдены координаты для оригинального изображения: {original_img}")
                    
            except Exception as e:
                print(f"Ошибка при вычислении координат для {patch_name}: {e}")
        
        if records:
            df = pd.DataFrame(records)
            
            # Создаем выходную папку если не существует
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем с координатами
            save_path = output_path / f"{city}_sliced_with_coords.csv"
            df.to_csv(save_path, index=False)
            
            print(f"Сохранено {len(df)} записей в {save_path}")
            return df
        else:
            print(f"Не удалось обработать ни одного патча для города {city}")
            return None
        
    except Exception as e:
        print(f"Ошибка при обработке координат: {e}")
        import traceback
        traceback.print_exc()
        return None