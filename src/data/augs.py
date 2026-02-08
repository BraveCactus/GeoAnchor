import io
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms


class RandomJPEGArtifacts:
    def __init__(self, p=0.5, quality_min=30, quality_max=80):
        self.p = p
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img):
        # иногда “портим” jpeg-ом
        if random.random() > self.p:
            return img
        q = random.randint(self.quality_min, self.quality_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianNoise:
    def __init__(self, p=0.5, sigma_min=1.0, sigma_max=8.0):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        # чуть-чуть шума накинуть
        if random.random() > self.p:
            return img
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        arr = np.array(img).astype(np.float32)
        arr += np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class RandomBlur:
    def __init__(self, p=0.5, radius_min=0.3, radius_max=1.5):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        # размытие, типа камера дернулась
        if random.random() > self.p:
            return img
        r = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


def build_transforms(image_size, mode):
    """
    mode:
      - stageA: simclr на орто-тайлах
      - query: под дрон (похуже/жестче)
      - tile: реф тайлы (почище)
      - eval: без рандома, просто детерминизм
    """
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    if mode == "eval":
        return transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            normalize,
        ])

    if mode == "stageA":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02
            ),
            RandomBlur(p=0.4),
            RandomJPEGArtifacts(p=0.3),
            RandomGaussianNoise(p=0.3),
            transforms.ToTensor(),
            normalize,
        ])

    if mode == "query":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03
            ),
            RandomBlur(p=0.5, radius_min=0.3, radius_max=2.0),
            RandomJPEGArtifacts(p=0.6, quality_min=20, quality_max=70),
            RandomGaussianNoise(p=0.5, sigma_min=2.0, sigma_max=10.0),
            transforms.ToTensor(),
            normalize,
        ])

    if mode == "tile":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
            normalize,
        ])

    raise ValueError(f"Unknown mode: {mode}")
