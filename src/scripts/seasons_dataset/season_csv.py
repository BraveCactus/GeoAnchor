import json
import pandas as pd
from pathlib import Path

from src.models.models import Dinov2EmbendingExtractor
from src.scripts.images_tools import utm26910_to_wgs84
from src.scripts.preprocess_images import extract_embedding_from_image

def create_season_csv(season='autumn', extractor=None, in_path=None, out_path=None):
    source_path = Path(in_path) / season    
    target_path = Path(out_path) / season
    
    target_path.mkdir(parents=True, exist_ok=True)

    with open("src/data/city_data.json", "r") as f:
        city_data = json.load(f)    
    
    american_coords = city_data["american_coordinates"]
    map_left = american_coords["left_bottom"]["easting"]
    map_bottom = american_coords["left_bottom"]["northing"]
    map_right = american_coords["right_top"]["easting"]
    map_top = american_coords["right_top"]["northing"]
    
    grid_size = 57
    total_size = map_right - map_left
    step = total_size / grid_size
    
    records = []    
    
    print(f"\n{'='*50}")
    print(f"Сезон: {season.upper()}")
    print(f"{'='*50}")    
    print(f"Сетка: {grid_size}x{grid_size} = {grid_size*grid_size} патчей")
    print(f"Размер патча: {step:.1f} м")
    print(f"{'='*50}\n")    
    
    total_patches = 0
    processed_patches = 0
    
    for i in range(grid_size):        
        row_patches = 0
        
        for j in range(grid_size):
            patch_name = f"austin_patch_{i}_{j}_{season}.tiff"
            patch_path = source_path / patch_name
            total_patches += 1
            
            if not patch_path.exists():
                print(f"  Патч не найден: {patch_path}")
                continue
            
            try:
                _, embedding = extract_embedding_from_image(str(patch_path), extractor)
                embedding_str = ';'.join(map(str, embedding))
            except Exception as e:
                print(f"  Ошибка обработки {patch_name}: {e}")
                embedding_str = ""            
            
            patch_left = map_left + j * step
            patch_right = patch_left + step
            
            patch_bottom = map_bottom + i * step
            patch_top = patch_bottom + step
            
            center_x = (patch_left + patch_right) / 2
            center_y = (patch_bottom + patch_top) / 2
            
            lat, lon = utm26910_to_wgs84(center_x, center_y)
            
            rel_x = (center_x - map_left) / total_size
            rel_y = (center_y - map_bottom) / total_size
            
            record = {
                "patch_id": i * grid_size + j,
                "patch_name": patch_name,
                "original_image": f"austin_map_{season}",
                "patch_coords": f"{i}_{j}",
                "grid_i": i,
                "grid_j": j,
                "tile_path": str(patch_path),
                "lat": lat,
                "lon": lon,
                "patch_left": patch_left,
                "patch_bottom": patch_bottom,
                "patch_right": patch_right,
                "patch_top": patch_top,
                "center_x": center_x,
                "center_y": center_y,
                "rel_x": rel_x,
                "rel_y": rel_y,
                "split": "train",
                "embedding": embedding_str
            }
            
            records.append(record)
            processed_patches += 1
            row_patches += 1        
       
        percent = (processed_patches / total_patches) * 100 if total_patches > 0 else 0
        
        print(f"  Строка {i+1:2d}/{grid_size} | "
              f"Найдено: {row_patches:2d} | "
              f"Всего: {processed_patches:4d}/{total_patches:4d} | "
              f"{percent:5.1f}%")
    
    df = pd.DataFrame(records)
    csv_path = out_path / f"austin_{season}_patches_with_embeddings.csv"
    df.to_csv(csv_path, index=False)    
    
    print(f"\n{'='*50}")
    print(f"ГОТОВО: {season.upper()}")
    print(f"  Записей: {len(df)}")
    print(f"  Файл: {csv_path}")   
    print(f"{'='*50}\n")
    
    return df

def create_all_seasons(seasons, in_path, out_path, extractor=None):  
    print("="*50)
    print("НАЧАЛО ОБРАБОТКИ ВСЕХ СЕЗОНОВ")
    print("="*50)
    
    for season in seasons:
        create_season_csv(season, extractor=extractor, in_path=in_path, out_path=out_path)
    
    print("="*50)
    print(f"Все CSV сохранены в: {out_path}/[сезон]/")
    print("ВСЕ СЕЗОНЫ ОБРАБОТАНЫ")
    print("="*50)

if __name__ == "__main__": 
    in_path = Path("results/seasonal_dataset")
    out_path = Path("results/seasonal_csv")

    seasons = ['autumn', 'winter']

    extractor = Dinov2EmbendingExtractor()

    create_all_seasons(seasons, in_path, out_path, extractor=extractor)