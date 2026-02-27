import numpy as np
import cv2
import tifffile as tiff
from pathlib import Path

def stitch_austin_maps():
    """Склеивает 3249 обработанных снимков Остина в одну большую карту"""    
    
    base_path = Path("results/seasonal_dataset")
    out_path = Path("results/stitched_seasons_austin")
    out_path.mkdir(parents=True, exist_ok=True)    
    
    grid_size = 57
    tile_size = 518
    full_size = grid_size * tile_size    
    
    for season in ['winter']:
        print(f"\nСклеивание {season}...")
        
        big_image = np.zeros((full_size, full_size, 3), dtype=np.uint8)
        count = 0
        
        for row in range(grid_size):
            for col in range(grid_size):
                filename = f"austin_patch_{row}_{col}_{season}.tiff"
                file_path = base_path / season / filename
                
                if file_path.exists():
                    img = cv2.imread(str(file_path))
                    
                    if img is not None:
                        y1 = row * tile_size
                        y2 = y1 + tile_size
                        x1 = col * tile_size
                        x2 = x1 + tile_size
                        
                        big_image[y1:y2, x1:x2] = img
                        count += 1
                        
            if row % 10 == 0:
                print(f"  Обработано строк: {row+1}/{grid_size}")
        
        print(f"  Загружено {count} файлов из {grid_size*grid_size}")
        
        if count > 0:
            out_file = out_path / f"austin_{season}_full.tiff"
            tiff.imwrite(str(out_file), big_image, compression='lzw')
            print(f"Сохранено: {out_file}")
            
            small_size = full_size // 4
            small = cv2.resize(big_image, (small_size, small_size), interpolation=cv2.INTER_LANCZOS4)
            small_file = out_path / f"austin_{season}_small.tiff"
            tiff.imwrite(str(small_file), small, compression='lzw')
            print(f"Сжато: {small_file}")
    
    print("\nГотово!")

if __name__ == "__main__":
    stitch_austin_maps()