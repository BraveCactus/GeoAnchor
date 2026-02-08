from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
import tifffile as tiff

from data.augs import build_transforms
from model.dinov2_gem import DinoV2Encoder


def resolve_path(p: str, base_dir: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.isabs(p) and not os.path.exists(p):
        return os.path.join(base_dir, os.path.basename(p))
    return os.path.join(base_dir, p)


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


def infer_tile_image_path(tiles_root: str, patch_name: str, exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")) -> str:
    base = os.path.join(tiles_root, patch_name)
    for ext in exts:
        p = base + ext
        if os.path.exists(p):
            return p
    return base + exts[0]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def save_meta(df: pd.DataFrame, out_meta: Path) -> None:
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_meta.suffix.lower()

    if suffix in {".sqlite", ".db"}:
        with sqlite3.connect(out_meta) as conn:
            df.to_sql("tile_meta", conn, if_exists="replace", index=False)
    elif suffix == ".parquet":
        df.to_parquet(out_meta, index=False)
    elif suffix == ".csv":
        df.to_csv(out_meta, index=False)
    else:
        raise ValueError(
            "Неподдерживаемый формат --out-meta. Используйте .sqlite, .parquet или .csv")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(
        description="Embed tile images with DinoV2Encoder ckpt and build FAISS index")
    ap.add_argument("--tiles_csv", required=True, type=Path,
                    help="CSV like StageA: patch_name,...,lat/lon(or x/y),split,...")
    ap.add_argument("--split", default="",
                    help="Filter by split value (e.g. train/val/test). Empty=all")

    ap.add_argument("--tiles_root", required=True, type=Path,
                    help="Folder with tile images named patch_name.tif (etc)")
    ap.add_argument("--image_col", default="",
                    help="If CSV has a column with image path, provide its name")

    ap.add_argument("--ckpt", required=True, type=Path,
                    help="stageA_best.pt / stageB_last.pt")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--out-index", type=Path,
                    default=Path("app/data/city.index.faiss"))
    ap.add_argument("--out-meta", type=Path,
                    default=Path("app/data/tile_meta.sqlite"))
    ap.add_argument("--out-vectors", type=Path,
                    default=Path("app/data/city.vectors.npy"))

    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize vectors before indexing (recommended for IP/cosine)")
    ap.add_argument(
        "--metric", choices=["ip", "l2"], default="ip", help="ip = cosine if normalized")
    args = ap.parse_args()

    tiles_root = args.tiles_root.expanduser().resolve()
    if not tiles_root.is_dir():
        raise FileNotFoundError(f"--tiles_root is not a dir: {tiles_root}")

    df = pd.read_csv(args.tiles_csv)

    if "patch_name" not in df.columns:
        raise ValueError("tiles_csv must contain column 'patch_name'")
    if "split" not in df.columns:
        raise ValueError("tiles_csv must contain column 'split'")

    has_latlon = {"lat", "lon"}.issubset(df.columns)
    has_xy = {"x", "y"}.issubset(df.columns)
    if not (has_latlon or has_xy):
        raise ValueError(
            "tiles_csv must contain either lat/lon or x/y columns")

    if args.split:
        df = df[df["split"].astype(str) == args.split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows after filtering split='{args.split}'")

    use_csv_path = args.image_col.strip() != "" and (args.image_col in df.columns)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tfm = build_transforms(args.image_size, mode="eval")

    model = DinoV2Encoder(backbone_name=args.backbone,
                          proj_dim=args.proj_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if ("model" in ckpt) else ckpt

    if any(k.startswith("enc.") for k in sd.keys()):
        sd = {k[len("enc."):]: v for k, v in sd.items()
              if k.startswith("enc.")}
    elif any(k.startswith("model.enc.") for k in sd.keys()):
        sd = {k[len("model.enc."):]: v for k, v in sd.items()
              if k.startswith("model.enc.")}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded encoder. missing:", len(
        missing), "unexpected:", len(unexpected))

    model.eval()

    img_paths: list[str] = []
    for r in df.itertuples(index=False):
        patch_name = str(getattr(r, "patch_name"))
        if use_csv_path:
            img_path = resolve_path(
                str(getattr(r, args.image_col)), str(tiles_root))
        else:
            img_path = infer_tile_image_path(str(tiles_root), patch_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Missing tile image: {img_path} (patch_name={patch_name})")
        img_paths.append(img_path)

    vectors = []
    for i in tqdm(range(0, len(img_paths), args.batch_size), desc="Embedding tiles"):
        batch_paths = img_paths[i:i + args.batch_size]
        imgs = []
        for p in batch_paths:
            img = open_image_any(p)
            imgs.append(tfm(img))
        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
        z = model(x)
        vectors.append(z.detach().cpu().numpy().astype("float32"))

    vectors_np = np.concatenate(vectors, axis=0)  # (N, D)
    if vectors_np.shape[1] != args.proj_dim:
        raise ValueError(
            f"Unexpected embed dim: got {vectors_np.shape[1]} expected {args.proj_dim}")

    if args.normalize:
        vectors_np = l2_normalize(vectors_np).astype("float32", copy=False)

    d = vectors_np.shape[1]
    if args.metric == "ip":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(vectors_np)

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    args.out_vectors.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out_index))
    np.save(args.out_vectors, vectors_np.astype("float32", copy=False))

    meta_cols = ["patch_name", "split"]
    if "image_name" in df.columns:
        meta_cols.append("image_name")
    if "tile_path" in df.columns:
        meta_cols.append("tile_path")
    if has_latlon:
        meta_cols += ["lat", "lon"]
    else:
        meta_cols += ["x", "y"]

    meta = df[meta_cols].copy().reset_index(drop=True)
    meta["faiss_id"] = meta.index.astype(int)

    save_meta(meta, args.out_meta)

    print(f"Saved index   -> {args.out_index}  (d={d}, ntotal={index.ntotal})")
    print(f"Saved vectors -> {args.out_vectors}  shape={vectors_np.shape}")
    print(f"Saved meta    -> {args.out_meta}  rows={len(meta)}")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
