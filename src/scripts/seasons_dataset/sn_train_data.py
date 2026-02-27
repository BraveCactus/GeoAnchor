import numpy as np
import cv2
from pathlib import Path

def create_train_season_data(seasons=['autumn'], num_images=3249, in_path=None, out_path=None):
    """Создает сезонный датасет, преобразуя изображения в соответствии с заданными сезонами и сохраняя их в новой папке
        Args:
            seasons (list): Список сезонов для создания датасетов
            num_images (int): Количество изображений для обработки
            in_path (str): Путь к папке с исходными изображениями
            out_path (str): Путь к папке для сохранения сезонного датасета
        Returns:
            None
    """
    source_path = Path(in_path)
    target_path = Path(out_path)
    
    target_path.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        (target_path / season).mkdir(exist_ok=True)
    
    tiff_files = list(source_path.glob("*.tiff")) + list(source_path.glob("*.tif"))
    tiff_files = tiff_files[:num_images]
    
    for i, img_file in enumerate(tiff_files):
        if i % 100 == 0:
            print(f"Обработано {i}/{len(tiff_files)}")
        
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]
        
        if 'autumn' in seasons:
            autumn = img.copy()
            hsv = cv2.cvtColor(autumn, cv2.COLOR_BGR2HSV)
            
            green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 30)
            
            hsv[:, :, 0] = np.where(green_mask, 20, hsv[:, :, 0])
            hsv[:, :, 1] = np.where(green_mask, 180, hsv[:, :, 1])
            
            autumn = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            warm = np.full((h, w, 3), [15, 10, 5], dtype=np.uint8)
            autumn = cv2.addWeighted(autumn, 0.85, warm, 0.15, 0)
            
            out_name = target_path / 'autumn' / f"{img_file.stem}_autumn.tiff"
            cv2.imwrite(str(out_name), autumn)
        
        if 'winter' in seasons:
            winter = img.copy()
            hsv = cv2.cvtColor(winter, cv2.COLOR_BGR2HSV)            
            
            water_mask = (hsv[:, :, 0] >= 75) & (hsv[:, :, 0] <= 90) & \
                        (hsv[:, :, 1] >= 30) & (hsv[:, :, 1] <= 70) & \
                        (hsv[:, :, 2] >= 60) & (hsv[:, :, 2] <= 110)
            
            green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & \
                        (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 40)            
            
            soil_mask = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 30) & \
                       (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 40)            
            
            dark = (hsv[:, :, 2] <= 35)            
            
            snow_mask = (green_mask | soil_mask) & ~dark & ~water_mask            
            
            hsv[:, :, 1] = np.where(snow_mask, hsv[:, :, 1] * 0.2, hsv[:, :, 1])
            hsv[:, :, 2] = np.where(snow_mask, np.clip(hsv[:, :, 2] * 1.1, 180, 210), hsv[:, :, 2])
            hsv[:, :, 0] = np.where(snow_mask, 0, hsv[:, :, 0])
            
            winter = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)            
            
            white = np.full_like(winter, 20)
            winter = np.where(snow_mask[:, :, np.newaxis], 
                            cv2.addWeighted(winter, 0.95, white, 0.05, 0),
                            winter)
            
            out_name = target_path / 'winter' / f"{img_file.stem}_winter.tiff"
            cv2.imwrite(str(out_name), winter)
    
    print(f"Создан сезонный датасет в {target_path}!")

if __name__ == "__main__":
    in_path = Path("results/sliced_images")
    out_path = Path("results/seasonal_dataset")
    create_train_season_data(
        in_path=in_path,
        out_path=out_path,
        num_images=3249,
        seasons=['autumn', 'winter']
    )