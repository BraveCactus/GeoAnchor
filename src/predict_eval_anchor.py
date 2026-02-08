import argparse
import json
import math
import os
import random
import sqlite3
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tifffile as tiff
from torchvision import transforms
from tqdm import tqdm

from model.dinov2_gem import DinoV2Encoder

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def sanitize_patch_name(name: str) -> str:
    return os.path.basename(str(name).replace("\\", "/"))


def infer_tile_path(tiles_root: str, patch_name: str, exts=IMAGE_EXTS) -> str:
    name = sanitize_patch_name(patch_name)
    base = os.path.join(tiles_root, name)

    if os.path.splitext(name)[1].lower() in exts and os.path.exists(base):
        return base

    for ext in exts:
        p = base + ext
        if os.path.exists(p):
            return p
    return base + exts[0]


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R * c


def inv_softplus(x: float) -> float:
    if x > 20:
        return x
    return float(math.log(math.expm1(max(x, 1e-8))))


def load_anchor_meta(meta_path: str):

    meta_path = os.path.abspath(meta_path)
    con = sqlite3.connect(meta_path)
    try:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)[
            "name"].tolist()
        if "anchor_meta" in tables:
            df = pd.read_sql("SELECT * FROM anchor_meta", con)
            id_col = "anchor_id" if "anchor_id" in df.columns else "faiss_id"
        elif "tile_meta" in tables:
            df = pd.read_sql("SELECT * FROM tile_meta", con)
            id_col = "faiss_id"
        else:
            raise ValueError(
                "SQLite meta must contain table 'anchor_meta' or 'tile_meta'")
    finally:
        con.close()

    need = [id_col, "lat", "lon"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"anchors_meta missing required column '{c}'")

    df = df.copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    if df[id_col].isna().any():
        raise ValueError("anchors_meta has non-numeric id values")

    df[id_col] = df[id_col].astype(np.int64)
    df = df.sort_values(id_col).reset_index(drop=True)

    ids = df[id_col].to_numpy(np.int64)
    exp = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(ids, exp):
        raise ValueError(
            f"{id_col} must be contiguous 0..N-1 (got min={ids.min()} max={ids.max()} len={len(ids)})")

    a_lat = df["lat"].astype(np.float64).to_numpy()
    a_lon = df["lon"].astype(np.float64).to_numpy()
    names = None
    if "tile_name" in df.columns:
        names = df["tile_name"].astype(str).to_numpy()
    elif "tile_name" in df.columns:
        names = df["tile_name"].astype(str).to_numpy()
    else:
        names = np.array([str(i) for i in range(len(df))], dtype=object)

    if not np.isfinite(a_lat).all() or not np.isfinite(a_lon).all():
        raise ValueError("anchors_meta contains non-finite lat/lon")

    return a_lat, a_lon, names


class AnchorModel(nn.Module):
    def __init__(self, backbone: str, proj_dim: int, learn_t: bool = False, init_t: float = 0.07):
        super().__init__()
        self.enc = DinoV2Encoder(backbone_name=backbone, proj_dim=proj_dim)
        self.adapter = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.learn_t = bool(learn_t)
        if self.learn_t:
            self.temp_raw = nn.Parameter(torch.tensor(
                inv_softplus(float(init_t)), dtype=torch.float32))
        else:
            self.register_parameter("temp_raw", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.adapter(z)
        z = F.normalize(z, p=2, dim=1)
        return z

    def get_temperature(self, fallback_t: float) -> torch.Tensor:
        if self.learn_t and self.temp_raw is not None:
            return F.softplus(self.temp_raw) + 1e-6
        return torch.tensor(float(fallback_t), device=self.adapter[0].weight.device)


def infer_learn_t_from_state_dict(sd: dict) -> bool:
    return any(k.endswith("temp_raw") for k in sd.keys())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sep", default="auto")

    ap.add_argument("--anchors_vectors", required=True,
                    help="anchors.vectors.npy (N,D) float32")
    ap.add_argument("--anchors_meta", required=True,
                    help="anchor_meta.sqlite with lat/lon (ids 0..N-1)")
    ap.add_argument("--ckpt", required=True, help="stageB_anchor_last.pt")

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--K", type=int, default=256,
                    help="topK by similarity if geo_knn is not provided")
    ap.add_argument("--geo_knn_npy", default="",
                    help="train_geo_knn.npy (N_train,Kgeo) int64 (anchor ids)")
    ap.add_argument("--Tpred", type=float, default=0.10,
                    help="temperature for prediction softmax")
    ap.add_argument("--Tsim", type=float, default=0.07,
                    help="fallback temperature for similarity (if model has learn_T)")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default="",
                    help="optional path to save metrics json")
    ap.add_argument("--out_csv", default="",
                    help="optional path to save per-sample csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: evaluate only first N samples (0=all)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.sep == "auto":
        df = pd.read_csv(args.tiles_csv, sep=None, engine="python")
    else:
        sep_real = "\t" if args.sep == "\\t" else args.sep
        df = pd.read_csv(args.tiles_csv, sep=sep_real)

    for c in ("patch_name", "split", "lat", "lon"):
        if c not in df.columns:
            raise ValueError(f"tiles_csv must contain '{c}'")

    df = df[df["split"].astype(str) == str(args.split)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for split='{args.split}'")
    if args.limit and args.limit > 0:
        df = df.iloc[: int(args.limit)].reset_index(drop=True)

    anchors = np.load(args.anchors_vectors).astype(np.float32)
    if anchors.ndim != 2 or anchors.shape[1] != int(args.proj_dim):
        raise ValueError(
            f"anchors_vectors must be (N,{args.proj_dim}), got {anchors.shape}")

    anchors = anchors / \
        np.clip(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-12, None)
    anchors_t = torch.from_numpy(anchors).to(device)  # (N,D)

    a_lat, a_lon, _ = load_anchor_meta(args.anchors_meta)
    if len(a_lat) != anchors.shape[0]:
        raise ValueError(
            f"anchors_meta rows != anchors_vectors rows: {len(a_lat)} vs {anchors.shape[0]}")
    a_lat_t = torch.from_numpy(a_lat.astype(np.float32)).to(device)
    a_lon_t = torch.from_numpy(a_lon.astype(np.float32)).to(device)

    geo_knn = None
    if str(args.geo_knn_npy).strip():
        geo_knn = np.load(args.geo_knn_npy)
        if geo_knn.ndim != 2:
            raise ValueError(
                f"--geo_knn_npy must be 2D (N,K), got {geo_knn.shape}")
        if geo_knn.shape[0] != len(df):
            raise ValueError(
                f"--geo_knn_npy first dim must match N_eval ({len(df)}), got {geo_knn.shape[0]}. "
                "Сгенерируй geo_knn именно для этого split."
            )
        if geo_knn.dtype not in (np.int64, np.int32):
            geo_knn = geo_knn.astype(np.int64)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise ValueError(
            "ckpt must be a dict state_dict or {'model': state_dict}")

    learn_t = infer_learn_t_from_state_dict(sd)
    model = AnchorModel(args.backbone, args.proj_dim,
                        learn_t=learn_t, init_t=float(args.Tsim)).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:10])
    if missing:
        print("[warn] missing keys:", missing[:10])

    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    dists = []
    rows = []

    bs = int(args.batch_size)
    n = len(df)

    for start in tqdm(range(0, n, bs), desc="Eval(anchor)"):
        end = min(n, start + bs)
        batch_df = df.iloc[start:end]

        imgs = []
        true_lat = batch_df["lat"].astype(np.float32).to_numpy()
        true_lon = batch_df["lon"].astype(np.float32).to_numpy()

        patch_names = batch_df["patch_name"].astype(str).to_list()
        for pn in patch_names:
            p = infer_tile_path(args.tiles_root, pn)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")
            imgs.append(tfm(open_image_any(p)))

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
        z = model(x)

        if geo_knn is not None:
            cand_ids_np = geo_knn[start:end]
            cand_ids = torch.from_numpy(
                cand_ids_np.astype(np.int64)).to(device)
            cand_anchors = anchors_t[cand_ids]
            sim = torch.einsum("bd,bkd->bk", z, cand_anchors)
            cand_lat = a_lat_t[cand_ids]
            cand_lon = a_lon_t[cand_ids]
        else:
            sim_full = z @ anchors_t.T
            K = min(int(args.K), sim_full.shape[1])
            sim, cand_ids = torch.topk(sim_full, k=K, dim=1, largest=True)
            cand_lat = a_lat_t[cand_ids]
            cand_lon = a_lon_t[cand_ids]

        w = F.softmax(sim / float(args.Tpred), dim=1)  # (B,K)
        pred_lat = (w * cand_lat).sum(dim=1).detach().cpu().numpy()
        pred_lon = (w * cand_lon).sum(dim=1).detach().cpu().numpy()

        dm = haversine_m(true_lat, true_lon, pred_lat.astype(
            np.float32), pred_lon.astype(np.float32))
        dists.extend(dm.tolist())

        for i in range(end - start):
            rows.append({
                "patch_name": patch_names[i],
                "true_lat": float(true_lat[i]),
                "true_lon": float(true_lon[i]),
                "pred_lat": float(pred_lat[i]),
                "pred_lon": float(pred_lon[i]),
                "dist_m": float(dm[i]),
            })

    d = np.array(dists, dtype=np.float64)
    out = {
        "n": int(len(d)),
        "split": str(args.split),
        "median_m": float(np.median(d)) if d.size else None,
        "mean_m": float(np.mean(d)) if d.size else None,
        "p90_m": float(np.quantile(d, 0.90)) if d.size else None,
        "p95_m": float(np.quantile(d, 0.95)) if d.size else None,
        "within_250m": float(np.mean(d <= 250.0)) if d.size else None,
        "within_500m": float(np.mean(d <= 500.0)) if d.size else None,
        "within_1000m": float(np.mean(d <= 1000.0)) if d.size else None,
        "within_2000m": float(np.mean(d <= 2000.0)) if d.size else None,
        "geo_knn_used": bool(geo_knn is not None),
        "K": int(geo_knn.shape[1]) if geo_knn is not None else int(min(args.K, anchors.shape[0])),
        "Tpred": float(args.Tpred),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(
            os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    if args.out_csv:
        os.makedirs(os.path.dirname(
            os.path.abspath(args.out_csv)), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
