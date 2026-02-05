import os

from src.data.images_tools import view_geotiff, show_tiff_image
from src.data.preprocess_images import load_and_preprocess_tiff

folder_path = r"inria_aerial_dataset\AerialImageDataset\train\images"
image_name = "austin1.tif"

if __name__ == "__main__":
    image_path = os.path.join(folder_path, image_name)
    load_and_preprocess_tiff(image_path)