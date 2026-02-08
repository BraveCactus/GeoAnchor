import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tifffile as tiff

from data.augs import build_transforms
from model.dinov2_gem import DinoV2Encoder


def open_image_any(path: str) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tiff.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in (3, 4) and arr.shape[1] > 16 and arr.shape[2] > 16:
                arr = arr[:3]
                arr = np.transpose(arr, (1, 2, 0))
            else:
                arr = arr[:, :, :3]
        else:
            raise ValueError(f"Unexpected TIFF shape {arr.shape} for {path}")

        if arr.dtype == np.uint16:
            arr = (arr / 65535.0 * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")
    return Image.open(path).convert("RGB")


def latlon_to_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat0r = np.deg2rad(lat0)
    x = (lon - lon0) * np.cos(lat0r) * 111320.0
    y = (lat - lat0) * 110540.0
    return x, y


class TestDataset(Dataset):
    def __init__(self, tiles_csv: str, split: str, tiles_root: str, image_size: int, sep: str,
                 lat0: float, lon0: float, exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        if sep == "auto":
            df = pd.read_csv(tiles_csv, sep=None, engine="python")
        else:
            sep_real = "\t" if sep == "\\t" else sep
            df = pd.read_csv(tiles_csv, sep=sep_real)

        if "split" not in df.columns:
            raise ValueError("CSV должен содержать колонку split")
        if "patch_name" not in df.columns:
            raise ValueError("CSV должен содержать patch_name")
        if not {"lat", "lon"}.issubset(df.columns):
            raise ValueError("CSV должен содержать lat и lon")

        df = df[df["split"].astype(str) == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Нет строк для split='{split}'")

        self.df = df
        self.tiles_root = os.path.abspath(tiles_root)
        self.exts = exts
        # мягко/детерминированно
        self.t = build_transforms(image_size, mode="tile")

        self.lat0 = float(lat0)
        self.lon0 = float(lon0)

        lat = df["lat"].astype(float).to_numpy()
        lon = df["lon"].astype(float).to_numpy()
        x, y = latlon_to_xy_m(lat, lon, self.lat0, self.lon0)
        self.x_m = x.astype(np.float32)
        self.y_m = y.astype(np.float32)

    def _infer_path(self, patch_name: str) -> str:
        base = os.path.join(self.tiles_root, patch_name)
        for ext in self.exts:
            p = base + ext
            if os.path.exists(p):
                return p
        return base + self.exts[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        patch_name = str(self.df.iloc[i]["patch_name"])
        p = self._infer_path(patch_name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Не найден файл: {p}")
        img = open_image_any(p)
        x = self.t(img)
        gt = np.array([self.x_m[i], self.y_m[i]], dtype=np.float32)
        return x, gt


class GeoRegModel(nn.Module):
    def __init__(self, backbone: str, proj_dim: int = 512):
        super().__init__()
        self.enc = DinoV2Encoder(backbone_name=backbone, proj_dim=proj_dim)
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, 2),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.head(z)  # normalized xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--sep", default="auto")
    ap.add_argument("--split", default="test")

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--origin_json", required=True)

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.origin_json, "r", encoding="utf-8") as f:
        o = json.load(f)

    lat0 = float(o["lat0"])
    lon0 = float(o["lon0"])
    mx = float(o["mx"])
    my = float(o["my"])
    sx = float(o["sx"])
    sy = float(o["sy"])

    ds = TestDataset(args.tiles_csv, args.split, args.tiles_root,
                     args.image_size, args.sep, lat0, lon0)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GeoRegModel(args.backbone, args.proj_dim).to(dev)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    preds_m = []
    gts_m = []

    with torch.no_grad():
        for x, gt in dl:
            x = x.to(dev, non_blocking=True)
            pred_n = model(x).cpu().numpy()  # normalized

            # denorm to meters
            pred = np.empty_like(pred_n, dtype=np.float32)
            pred[:, 0] = pred_n[:, 0] * sx + mx
            pred[:, 1] = pred_n[:, 1] * sy + my

            preds_m.append(pred)
            gts_m.append(gt.numpy())

    preds_m = np.concatenate(preds_m, axis=0)
    gts_m = np.concatenate(gts_m, axis=0)

    err = np.sqrt(((preds_m - gts_m) ** 2).sum(axis=1))

    summary = {
        "n": int(err.shape[0]),
        "median_m": float(np.median(err)),
        "mean_m": float(err.mean()),
        "p90_m": float(np.quantile(err, 0.90)),
        "p95_m": float(np.quantile(err, 0.95)),
        "within_250m": float((err <= 250).mean()),
        "within_500m": float((err <= 500).mean()),
        "within_1000m": float((err <= 1000).mean()),
        "within_2000m": float((err <= 2000).mean()),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
