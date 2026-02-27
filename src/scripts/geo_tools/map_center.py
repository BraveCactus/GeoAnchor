import rasterio
import os
import glob
from pyproj import Transformer

def utm26914_to_wgs84(x, y):
    """Конвертация из UTM NAD83 в WGS84"""
    transformer = Transformer.from_crs("EPSG:26914", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


folder = r'C:\Users\matve\Desktop\GeoAnchor\inria_aerial_dataset\AerialImageDataset\train\images'

files = glob.glob(os.path.join(folder, 'austin*.tif'))

min_lon = float('inf')
min_lat = float('inf')
max_lon = float('-inf')
max_lat = float('-inf')

min_easting = float('inf')
min_northing = float('inf')
max_easting = float('-inf')
max_northing = float('-inf')

for file in files:
    with rasterio.open(file) as src:
        left, bottom, right, top = src.bounds
        
        lon_left, lat_bottom = utm26914_to_wgs84(left, bottom)
        lon_right, lat_top = utm26914_to_wgs84(right, top)
        
        min_lon = min(min_lon, lon_left)
        min_lat = min(min_lat, lat_bottom)
        max_lon = max(max_lon, lon_right)
        max_lat = max(max_lat, lat_top)
        
        min_easting = min(min_easting, left)
        min_northing = min(min_northing, bottom)
        max_easting = max(max_easting, right)
        max_northing = max(max_northing, top)

result_wgs84 = {
    'left_bottom': (min_lat, min_lon),      
    'right_top': (max_lat, max_lon),       
    'center': ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
}

result_utm = {
    'left_bottom': (min_easting, min_northing),    
    'right_top': (max_easting, max_northing),      
    'center': ((min_easting + max_easting) / 2, (min_northing + max_northing) / 2)
}

print("=== WGS84 (градусы) ===")
print(f"Левый нижний:  lat={result_wgs84['left_bottom'][0]:.6f}, lon={result_wgs84['left_bottom'][1]:.6f}")
print(f"Правый верхний: lat={result_wgs84['right_top'][0]:.6f}, lon={result_wgs84['right_top'][1]:.6f}")
print(f"Центр:         lat={result_wgs84['center'][0]:.6f}, lon={result_wgs84['center'][1]:.6f}")

print("\n=== UTM NAD83 / зона 14N (метры) ===")
print(f"Левый нижний:  easting={result_utm['left_bottom'][0]:.1f}, northing={result_utm['left_bottom'][1]:.1f}")
print(f"Правый верхний: easting={result_utm['right_top'][0]:.1f}, northing={result_utm['right_top'][1]:.1f}")
print(f"Центр:         easting={result_utm['center'][0]:.1f}, northing={result_utm['center'][1]:.1f}")

width = max_easting - min_easting
height = max_northing - min_northing
print(f"\n=== Размер области ===")
print(f"Ширина: {width:.1f} м ({width/1000:.2f} км)")
print(f"Высота: {height:.1f} м ({height/1000:.2f} км)")