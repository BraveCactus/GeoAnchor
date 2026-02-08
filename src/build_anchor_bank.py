import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
import tifffile as tiff
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.augs import build_transforms
from model.dinov2_gem import DinoV2Encoder

IMAGE_EXTS: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


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
                if arr.shape[2] < 3:
                    raise ValueError(
                        f"TIFF has <3 channels: {arr.shape} for {path}")
                arr = arr[:, :, :3]
        else:
            raise ValueError(f"Unexpected TIFF shape: {arr.shape} for {path}")

        if arr.dtype == np.uint16:
            arr = (arr / 65535.0 * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    return Image.open(path).convert("RGB")


def sanitize_patch_name(patch_name: str) -> str:
    return os.path.basename(str(patch_name).replace("\\", "/"))


def infer_tile_path(tiles_root: str, patch_name: str, exts: Tuple[str, ...] = IMAGE_EXTS) -> str:
    patch_name = sanitize_patch_name(patch_name)
    base = os.path.join(tiles_root, patch_name)

    if os.path.splitext(patch_name)[1].lower() in exts and os.path.exists(base):
        return base

    for ext in exts:
        p = base + ext
        if os.path.exists(p):
            return p

    return base + exts[0]


class AnchorTilesDataset(Dataset):
    def __init__(self, tiles_csv: str, tiles_root: str, split: str, image_size: int, sep: str = "auto"):
        if sep == "auto":
            df = pd.read_csv(tiles_csv, sep=None, engine="python")
        else:
            sep_real = "\t" if sep == "\\t" else sep
            df = pd.read_csv(tiles_csv, sep=sep_real)

        for col in ("patch_name", "split", "lat", "lon"):
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column")

        df = df[df["split"].astype(str) == str(split)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows found for split='{split}'")

        self.df = df
        self.tiles_root = os.path.abspath(tiles_root)
        self.tfm = build_transforms(image_size, mode="eval")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patch_name = str(row["patch_name"])
        tile_name = sanitize_patch_name(patch_name)
        img_path = infer_tile_path(self.tiles_root, tile_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Tile image not found: {img_path} (patch_name={patch_name})")

        img = open_image_any(img_path)
        x = self.tfm(img)
        lat = np.float32(row["lat"])
        lon = np.float32(row["lon"])
        return x, tile_name, lat, lon


def load_encoder_weights(model: DinoV2Encoder, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if not isinstance(sd, dict):
        raise ValueError(
            "Checkpoint must be a state_dict or dict with key 'model'")

    if any(k.startswith("enc.") for k in sd.keys()):
        sd = {k[len("enc."):]: v for k, v in sd.items()
              if k.startswith("enc.")}
    elif any(k.startswith("model.enc.") for k in sd.keys()):
        sd = {k[len("model.enc."):]: v for k, v in sd.items()
              if k.startswith("model.enc.")}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys while loading encoder: first={unexpected[:5]}")
    if missing:
        print(f"[warn] Missing keys while loading encoder: {len(missing)}")


def save_anchor_meta_sqlite(tile_names: list[str], lats: np.ndarray, lons: np.ndarray, out_meta: str) -> None:
    out_path = Path(out_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(out_path))
    try:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS anchor_meta")
        cur.execute(
            """
            CREATE TABLE anchor_meta (
                anchor_id INTEGER PRIMARY KEY,
                tile_name TEXT,
                lat REAL,
                lon REAL
            )
            """
        )
        rows = [(int(i), str(tile_names[i]), float(lats[i]), float(lons[i]))
                for i in range(len(tile_names))]
        cur.executemany(
            "INSERT INTO anchor_meta(anchor_id, tile_name, lat, lon) VALUES (?, ?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build anchor bank vectors and metadata from train tiles")
    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sep", default="auto")

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument("--out_vectors", required=True)
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = AnchorTilesDataset(
        tiles_csv=args.tiles_csv,
        tiles_root=args.tiles_root,
        split=args.split,
        image_size=args.image_size,
        sep=args.sep,
    )
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True)

    model = DinoV2Encoder(backbone_name=args.backbone,
                          proj_dim=args.proj_dim).to(device)
    load_encoder_weights(model, args.ckpt)
    model.eval()

    vectors_chunks = []
    tile_names: list[str] = []
    lats: list[float] = []
    lons: list[float] = []

    with torch.no_grad():
        for x, names, lat, lon in tqdm(dl, desc="Embedding anchors"):
            x = x.to(device, non_blocking=True)
            z = model(x).detach().cpu().numpy().astype(np.float32)
            vectors_chunks.append(z)

            tile_names.extend([str(n) for n in names])
            lats.extend([float(v) for v in lat.numpy().tolist()])
            lons.extend([float(v) for v in lon.numpy().tolist()])

    vectors = np.concatenate(vectors_chunks, axis=0).astype(
        np.float32, copy=False)
    if vectors.ndim != 2 or vectors.shape[1] != int(args.proj_dim):
        raise ValueError(
            f"Unexpected vectors shape: {vectors.shape}, expected [N,{args.proj_dim}]")

    if args.normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)

    out_vectors = Path(args.out_vectors)
    out_vectors.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_vectors, vectors.astype(np.float32, copy=False))

    lat_arr = np.array(lats, dtype=np.float64)
    lon_arr = np.array(lons, dtype=np.float64)
    save_anchor_meta_sqlite(tile_names, lat_arr, lon_arr, args.out_meta)

    stats = {
        "n": int(vectors.shape[0]),
        "d": int(vectors.shape[1]),
        "split": str(args.split),
        "tiles_root": os.path.abspath(args.tiles_root),
        "lat_min": float(np.min(lat_arr)),
        "lat_max": float(np.max(lat_arr)),
        "lon_min": float(np.min(lon_arr)),
        "lon_max": float(np.max(lon_arr)),
        "normalized": bool(args.normalize),
        "out_vectors": str(out_vectors),
        "out_meta": str(Path(args.out_meta)),
    }

    stats_path = out_vectors.with_suffix(out_vectors.suffix + ".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
