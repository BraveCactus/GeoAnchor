import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import tifffile as tiff
from pathlib import Path
import random
import cv2
import pandas as pd
import json
from PIL import Image

# Отключаем защиту от decompression bomb
Image.MAX_IMAGE_PIXELS = None

from src.scripts.images_tools import utm26910_to_wgs84
from src.scripts.preprocess_images import extract_embedding_from_image
from src.models.models import Dinov2EmbendingExtractor

def load_city_bounds(city="austin"):
    json_path = Path("src/data/city_data.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    amer_coords = data["american_coordinates"]
    map_left = amer_coords["left_bottom"]["easting"]
    map_bottom = amer_coords["left_bottom"]["northing"]
    map_right = amer_coords["right_top"]["easting"]
    map_top = amer_coords["right_top"]["northing"]
    
    return map_left, map_bottom, map_right, map_top

def extract_and_save_patches(image_path, output_dir, num_patches=100, patch_size=518):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем через PIL с отключенной защитой
    img = Image.open(image_path)
    img = np.array(img)
    h, w = img.shape[:2]
    
    patches_info = []
    for i in range(num_patches):
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        
        patch = img[y:y+patch_size, x:x+patch_size].copy()
        
        angle = random.uniform(0, 360)
        center = (patch_size // 2, patch_size // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(patch, matrix, (patch_size, patch_size), borderMode=cv2.BORDER_REFLECT)
        
        filename = f"patch_{i:03d}_x{x}_y{y}_a{int(angle)}.tiff"
        filepath = output_dir / filename
        tiff.imwrite(str(filepath), rotated, compression=None)
        
        patches_info.append({
            'filename': filename,
            'filepath': filepath,
            'x': x,
            'y': y,
            'angle': angle,
            'patch_size': patch_size,
            'img_w': w,
            'img_h': h
        })
    
    return patches_info

def extract_embeddings_for_patches(patches_info, extractor=None):
    if extractor is None:
        extractor = Dinov2EmbendingExtractor()
    
    embeddings = []
    for info in patches_info:
        try:
            _, embedding = extract_embedding_from_image(str(info['filepath']), extractor)
            embedding_str = ';'.join(map(str, embedding))
        except:
            embedding_str = ""
        
        embeddings.append(embedding_str)
    
    return embeddings

def create_patches_csv(patches_info, embeddings, map_bounds, output_dir, city="austin"):
    map_left, map_bottom, map_right, map_top = map_bounds
    records = []
    
    for i, (info, emb) in enumerate(zip(patches_info, embeddings)):
        w, h = info['img_w'], info['img_h']
        patch_size = info['patch_size']
        
        px = info['x'] + patch_size // 2
        py = info['y'] + patch_size // 2
        
        meter_x = (map_right - map_left) / w
        meter_y = (map_top - map_bottom) / h
        
        cx = map_left + px * meter_x
        cy = map_top - py * meter_y
        
        lat, lon = utm26910_to_wgs84(cx, cy)
        
        rel_x = (cx - map_left) / (map_right - map_left)
        rel_y = (cy - map_bottom) / (map_top - map_bottom)
        
        records.append({
            'patch_id': i,
            'patch_name': info['filename'],
            'original_image': f"{city}_stitched",
            'patch_coords': f"{info['y']}_{info['x']}",
            'tile_path': str(info['filepath']),
            'lat': lat,
            'lon': lon,
            'center_x_utm': cx,
            'center_y_utm': cy,
            'patch_left': map_left + info['x'] * meter_x,
            'patch_bottom': map_top - (info['y'] + patch_size) * meter_y,
            'patch_right': map_left + (info['x'] + patch_size) * meter_x,
            'patch_top': map_top - info['y'] * meter_y,
            'rel_x': rel_x,
            'rel_y': rel_y,
            'split': 'val',
            'embedding': emb
        })
    
    df = pd.DataFrame(records)
    csv_path = output_dir / "rotated_patches_with_embeddings.csv"
    df.to_csv(csv_path, index=False)
    
    return df

def create_rotated_patches_dataset(image_path, output_dir, city="austin", num_patches=100, patch_size=518):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    map_bounds = load_city_bounds(city)
    
    patches_info = extract_and_save_patches(image_path, output_dir, num_patches, patch_size)
    
    embeddings = extract_embeddings_for_patches(patches_info)
    
    df = create_patches_csv(patches_info, embeddings, map_bounds, output_dir, city)
    
    return df

if __name__ == "__main__":
    in_path = Path("results/maps/austin_stitched_from_patches.tiff")
    out_path = Path("results/rotated_patches")
    
    df = create_rotated_patches_dataset(in_path, out_path, city="austin", num_patches=100)
    print(f"Готово! Создано {len(df)} патчей")