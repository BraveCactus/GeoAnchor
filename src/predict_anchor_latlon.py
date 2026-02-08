
import argparse
import json
import os
import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tifffile as tiff
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.dinov2_gem import DinoV2Encoder

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


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


def haversine_m_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371000.0
    p1 = np.deg2rad(lat1.astype(np.float64))
    p2 = np.deg2rad(lat2.astype(np.float64))
    dphi = np.deg2rad((lat2 - lat1).astype(np.float64))
    dl = np.deg2rad((lon2 - lon1).astype(np.float64))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * \
        np.cos(p2) * np.sin(dl / 2.0) ** 2
    return (2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float64)


def load_anchor_meta(meta_path: str):
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

    for c in (id_col, "lat", "lon"):
        if c not in df.columns:
            raise ValueError(f"anchors_meta missing required column '{c}'")

    df = df.copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    if df[id_col].isna().any():
        raise ValueError("anchors_meta has non-numeric ids")
    df[id_col] = df[id_col].astype(np.int64)
    df = df.sort_values(id_col).reset_index(drop=True)

    ids = df[id_col].to_numpy(dtype=np.int64)
    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError(f"{id_col} in anchors_meta must be contiguous 0..N-1")

    lat = df["lat"].astype(np.float32).to_numpy()
    lon = df["lon"].astype(np.float32).to_numpy()
    return lat, lon


class AnchorModel(nn.Module):
    def __init__(self, backbone: str, proj_dim: int, use_xy_head: bool):
        super().__init__()
        self.enc = DinoV2Encoder(backbone_name=backbone, proj_dim=proj_dim)
        self.adapter = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.use_xy_head = bool(use_xy_head)
        self.xy_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, 2),
        ) if self.use_xy_head else None

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        z = self.adapter(z)
        z = F.normalize(z, p=2, dim=1)
        xy = self.xy_head(z) if self.xy_head is not None else None
        return z, xy


def load_model_from_ckpt(model: AnchorModel, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if (isinstance(ckpt, dict)
                           and "model" in ckpt) else ckpt
    if not isinstance(sd, dict):
        raise ValueError(
            "Checkpoint must be state_dict or dict with key 'model'")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys while loading model: {unexpected[:5]}")
    if missing:
        print(f"[warn] Missing keys while loading model: {len(missing)}")


class CsvEvalDataset(Dataset):
    def __init__(self, tiles_csv: str, split: str, tiles_root: str, image_size: int, sep: str = "auto"):
        if sep == "auto":
            df = pd.read_csv(tiles_csv, sep=None, engine="python")
        else:
            sep_real = "\t" if sep == "\\t" else sep
            df = pd.read_csv(tiles_csv, sep=sep_real)

        for c in ("patch_name", "split", "lat", "lon"):
            if c not in df.columns:
                raise ValueError(f"tiles_csv must contain '{c}'")

        df = df[df["split"].astype(str) == str(split)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows for split='{split}'")

        self.df = df
        self.tiles_root = os.path.abspath(tiles_root)
        self.t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patch_name = str(row["patch_name"])
        p = infer_tile_path(self.tiles_root, patch_name)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Image not found for patch_name='{patch_name}'. Checked path: {p}")

        x = self.t(open_image_any(p))
        lat = np.float32(row["lat"])
        lon = np.float32(row["lon"])
        return x, patch_name, lat, lon, p


def predict_from_embeddings(
    z: torch.Tensor,
    anchors_t: torch.Tensor,
    a_lat_t: torch.Tensor,
    a_lon_t: torch.Tensor,
    top_m: int,
    tpred: float,
):
    sim = z @ anchors_t.T
    sim_top, idx_top = torch.topk(sim, k=top_m, dim=1, largest=True)
    w = F.softmax(sim_top / tpred, dim=1)

    pred_lat = (w * a_lat_t[idx_top]).sum(dim=1)
    pred_lon = (w * a_lon_t[idx_top]).sum(dim=1)
    return pred_lat, pred_lon, idx_top, w


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Anchor-based lat/lon prediction")
    ap.add_argument("--image_path", default="")
    ap.add_argument("--tiles_csv", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--sep", default="auto")

    ap.add_argument("--anchors_vectors", required=True)
    ap.add_argument("--anchors_meta", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--topM", type=int, default=128)
    ap.add_argument("--Tpred", type=float, default=0.10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--show_topk", type=int, default=5)
    args = ap.parse_args()

    if not args.image_path and not args.tiles_csv:
        raise ValueError("Provide either --image_path or --tiles_csv")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    anchors = np.load(args.anchors_vectors).astype(np.float32)
    if anchors.ndim != 2 or anchors.shape[1] != int(args.proj_dim):
        raise ValueError(
            f"anchors_vectors must be [N,{args.proj_dim}], got {anchors.shape}")

    lat_np, lon_np = load_anchor_meta(args.anchors_meta)
    if anchors.shape[0] != lat_np.shape[0]:
        raise ValueError(
            f"anchors_vectors rows != anchors_meta rows: {anchors.shape[0]} vs {lat_np.shape[0]}"
        )

    norms = np.linalg.norm(anchors, axis=1, keepdims=True)
    anchors = anchors / np.clip(norms, 1e-12, None)

    anchors_t = torch.from_numpy(anchors).to(device)
    a_lat_t = torch.from_numpy(lat_np).to(device)
    a_lon_t = torch.from_numpy(lon_np).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    use_xy_head = bool(ckpt_args.get("use_xy_head", False))

    model = AnchorModel(args.backbone, args.proj_dim,
                        use_xy_head=use_xy_head).to(device)
    load_model_from_ckpt(model, args.ckpt)
    model.eval()

    top_m = min(max(int(args.topM), 1), anchors.shape[0])
    show_topk = min(max(int(args.show_topk), 0), top_m)

    t = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    if args.image_path:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(
                f"image_path does not exist: {args.image_path}")

        x = t(open_image_any(args.image_path)).unsqueeze(0).to(device)
        z, _ = model(x)

        pred_lat, pred_lon, idx_top, w = predict_from_embeddings(
            z=z,
            anchors_t=anchors_t,
            a_lat_t=a_lat_t,
            a_lon_t=a_lon_t,
            top_m=top_m,
            tpred=float(args.Tpred),
        )

        out = {
            "pred_lat": float(pred_lat[0].item()),
            "pred_lon": float(pred_lon[0].item()),
            "top_ids": [int(v) for v in idx_top[0, :show_topk].cpu().numpy().tolist()],
            "top_weights": [float(v) for v in w[0, :show_topk].cpu().numpy().tolist()],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    ds = CsvEvalDataset(
        tiles_csv=args.tiles_csv,
        split=args.split,
        tiles_root=args.tiles_root,
        image_size=args.image_size,
        sep=args.sep,
    )
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True)

    rows = []
    dists = []

    for x, patch_name, lat_gt, lon_gt, img_path in dl:
        x = x.to(device, non_blocking=True)
        z, _ = model(x)

        pred_lat, pred_lon, idx_top, w = predict_from_embeddings(
            z=z,
            anchors_t=anchors_t,
            a_lat_t=a_lat_t,
            a_lon_t=a_lon_t,
            top_m=top_m,
            tpred=float(args.Tpred),
        )

        pred_lat_np = pred_lat.cpu().numpy().astype(np.float64)
        pred_lon_np = pred_lon.cpu().numpy().astype(np.float64)
        gt_lat_np = lat_gt.numpy().astype(np.float64)
        gt_lon_np = lon_gt.numpy().astype(np.float64)

        dist = haversine_m_np(gt_lat_np, gt_lon_np, pred_lat_np, pred_lon_np)
        dists.append(dist)

        for i in range(x.shape[0]):
            rows.append({
                "patch_name": str(patch_name[i]),
                "image_path": str(img_path[i]),
                "gt_lat": float(gt_lat_np[i]),
                "gt_lon": float(gt_lon_np[i]),
                "pred_lat": float(pred_lat_np[i]),
                "pred_lon": float(pred_lon_np[i]),
                "dist_m": float(dist[i]),
                "top_ids": json.dumps([int(v) for v in idx_top[i, :show_topk].cpu().numpy().tolist()], ensure_ascii=False),
                "top_weights": json.dumps([float(v) for v in w[i, :show_topk].cpu().numpy().tolist()], ensure_ascii=False),
            })

    d = np.concatenate(dists, axis=0) if dists else np.zeros(
        (0,), dtype=np.float64)

    summary = {
        "n": int(d.shape[0]),
        "median_m": float(np.median(d)) if d.size else float("nan"),
        "mean_m": float(np.mean(d)) if d.size else float("nan"),
        "p90_m": float(np.quantile(d, 0.90)) if d.size else float("nan"),
        "p95_m": float(np.quantile(d, 0.95)) if d.size else float("nan"),
        "within_250m": float((d <= 250.0).mean()) if d.size else float("nan"),
        "within_500m": float((d <= 500.0).mean()) if d.size else float("nan"),
        "within_1000m": float((d <= 1000.0).mean()) if d.size else float("nan"),
        "within_2000m": float((d <= 2000.0).mean()) if d.size else float("nan"),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
