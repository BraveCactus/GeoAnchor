
import argparse
import json
import os
import sqlite3
from typing import Tuple

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tifffile as tiff

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
            raise ValueError(f"Unexpected TIFF shape {arr.shape} for {path}")

        if arr.dtype == np.uint16:
            arr = (arr / 65535.0 * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    return Image.open(path).convert("RGB")


def load_anchor_meta(path: str):
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql(
            "SELECT anchor_id, tile_name, lat, lon FROM anchor_meta ORDER BY anchor_id", con)
    finally:
        con.close()

    if df.empty:
        raise ValueError("anchor_meta is empty")

    ids = df["anchor_id"].astype(np.int64).to_numpy()
    if not np.array_equal(ids, np.arange(len(df), dtype=np.int64)):
        raise ValueError("anchor_meta.anchor_id must be contiguous 0..N-1")

    names = df["tile_name"].astype(str).tolist()
    lat = df["lat"].astype(np.float32).to_numpy()
    lon = df["lon"].astype(np.float32).to_numpy()
    return lat, lon, names


class AnchorStageBModel(nn.Module):
    def __init__(self, backbone: str, proj_dim: int):
        super().__init__()
        self.enc = DinoV2Encoder(backbone_name=backbone, proj_dim=proj_dim)
        self.projector = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.projector(z)
        return z


def load_model_weights(model: AnchorStageBModel, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if not isinstance(sd, dict):
        raise ValueError(
            "Checkpoint must be a state_dict or dict with key 'model'")

    if any(k.startswith("enc.") for k in sd.keys()) and not any(k.startswith("projector.") for k in sd.keys()):
        missing, unexpected = model.enc.load_state_dict(
            {k[len("enc."):]: v for k, v in sd.items() if k.startswith("enc.")}, strict=False
        )
        if unexpected:
            raise ValueError(
                f"Unexpected keys while loading encoder fallback: first={unexpected[:5]}")
        if missing:
            print(
                f"[warn] Missing encoder keys in fallback load: {len(missing)}")
        print(
            "[warn] Loaded encoder-only checkpoint; projector remains randomly initialized.")
        return

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys while loading model: first={unexpected[:5]}")
    if missing:
        print(f"[warn] Missing model keys while loading: {len(missing)}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Predict center coordinates from one image using anchor bank")
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--tiles_root", default="",
                    help="Unused, kept for CLI compatibility")

    ap.add_argument("--anchors_vectors", required=True)
    ap.add_argument("--anchors_meta", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--topM", type=int, default=128)
    ap.add_argument("--Tpred", type=float, default=0.10)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(
            f"Input image does not exist: {args.image_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vec = np.load(args.anchors_vectors).astype(np.float32)
    if vec.ndim != 2:
        raise ValueError(f"anchors_vectors must be 2D, got shape={vec.shape}")

    lat, lon, _ = load_anchor_meta(args.anchors_meta)
    if vec.shape[0] != lat.shape[0]:
        raise ValueError(
            f"vectors/meta size mismatch: vectors={vec.shape[0]} meta={lat.shape[0]}")

    if args.normalize:
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / np.clip(norms, 1e-12, None)

    index = faiss.IndexFlatIP(vec.shape[1])
    index.add(vec)

    vec_t = torch.from_numpy(vec).to(device=device, dtype=torch.float32)
    lat_t = torch.from_numpy(lat).to(device=device, dtype=torch.float32)
    lon_t = torch.from_numpy(lon).to(device=device, dtype=torch.float32)

    model = AnchorStageBModel(backbone=args.backbone,
                              proj_dim=args.proj_dim).to(device)
    load_model_weights(model, args.ckpt)
    model.eval()

    tfm = build_transforms(args.image_size, mode="eval")
    img = open_image_any(args.image_path)
    x = tfm(img).unsqueeze(0).to(device)

    z = model(x)
    if args.normalize:
        z = F.normalize(z, p=2, dim=1)

    z_np = z.detach().cpu().numpy().astype(np.float32)
    top_m = int(min(max(1, args.topM), vec.shape[0]))
    scores, ids = index.search(z_np, top_m)

    top_ids = ids[0].astype(np.int64)
    top_scores = scores[0].astype(np.float32)

    ids_t = torch.from_numpy(top_ids).to(device)
    sim = (z * vec_t[ids_t]).sum(dim=1)
    w = F.softmax(sim / float(args.Tpred), dim=0)

    pred_lat = float((w * lat_t[ids_t]).sum().item())
    pred_lon = float((w * lon_t[ids_t]).sum().item())

    out = {
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "topM_ids": [int(v) for v in top_ids.tolist()],
        "topM_scores": [float(v) for v in top_scores.tolist()],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
