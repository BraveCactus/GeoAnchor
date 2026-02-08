import os
import json
import argparse
import shutil
import sqlite3
import math
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import tifffile as tiff
import faiss

from data.augs import build_transforms
from model.dinov2_gem import DinoV2Encoder


def open_image_any(path: str) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tiff.imread(path)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            # CHW
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


def infer_tile_path(tiles_root: str, patch_name: str,
                    exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")) -> str:
    patch_name = str(patch_name)
    patch_name = patch_name.replace("\\", "/")
    patch_name = os.path.basename(patch_name)

    base = os.path.join(tiles_root, patch_name)
    for ext in exts:
        p = base + ext
        if os.path.exists(p):
            return p
    # fallback
    return base + exts[0]


def safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    s = str(s)
    for c in bad:
        s = s.replace(c, "_")
    s = s.strip().strip(".")
    return s or "query"


def copy_or_symlink(src: str, dst: str, mode: str = "symlink"):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        if not os.path.exists(dst):
            os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def dist_xy(x1, y1, x2, y2) -> float:
    dx = float(x1) - float(x2)
    dy = float(y1) - float(y2)
    return float((dx * dx + dy * dy) ** 0.5)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * \
        math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))


def load_meta(meta_path: str) -> pd.DataFrame:
    p = meta_path.lower()
    if p.endswith(".sqlite") or p.endswith(".db"):
        with sqlite3.connect(meta_path) as conn:
            return pd.read_sql("SELECT * FROM tile_meta", conn)
    if p.endswith(".csv"):
        return pd.read_csv(meta_path)
    if p.endswith(".parquet"):
        return pd.read_parquet(meta_path)
    raise ValueError("Unsupported meta format. Use .sqlite/.db/.csv/.parquet")


def build_id2coord(meta: pd.DataFrame) -> Tuple[Dict[int, Tuple[float, float]], str]:

    if "faiss_id" not in meta.columns:
        raise ValueError("tile_meta must contain column 'faiss_id'")

    has_latlon = {"lat", "lon"}.issubset(meta.columns)
    has_xy = {"x", "y"}.issubset(meta.columns)

    if has_latlon:
        id2coord = {int(r.faiss_id): (float(r.lat), float(r.lon))
                    for r in meta.itertuples(index=False)}
        return id2coord, "latlon"
    if has_xy:
        id2coord = {int(r.faiss_id): (float(r.x), float(r.y))
                    for r in meta.itertuples(index=False)}
        return id2coord, "xy"

    raise ValueError("tile_meta must contain either lat/lon or x/y columns")


def build_id2img(meta: pd.DataFrame, index_tiles_root: str,
                 exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")) -> Dict[int, str]:

    if "faiss_id" not in meta.columns:
        raise ValueError("tile_meta must contain column 'faiss_id'")

    id2img: Dict[int, str] = {}

    name_col = None
    for c in ["patch_name", "tile_name"]:
        if c in meta.columns:
            name_col = c
            break

    path_col = None
    for c in ["tile_path", "path", "img_path", "image_path", "file_path", "filepath"]:
        if c in meta.columns:
            path_col = c
            break

    for r in meta.itertuples(index=False):
        fid = int(getattr(r, "faiss_id"))

        if name_col is not None:
            nm = str(getattr(r, name_col))
            p = infer_tile_path(index_tiles_root, nm, exts=exts)
            id2img[fid] = p
            continue

        if path_col is not None:
            raw = str(getattr(r, path_col))
            raw = raw.replace("\\", "/")
            base = os.path.basename(raw)
            p = os.path.join(index_tiles_root, base) if base else raw
            if os.path.isabs(raw) and os.path.exists(raw):
                p = raw
            id2img[fid] = p
            continue

        id2img[fid] = ""

    return id2img


def get_true_coord_from_csv(row, coord_kind: str) -> Tuple[float, float]:

    if coord_kind == "latlon":
        if not (hasattr(row, "lat") and hasattr(row, "lon")):
            raise ValueError(
                "tiles_csv must contain 'lat' and 'lon' columns for distance eval")
        return float(getattr(row, "lat")), float(getattr(row, "lon"))
    else:
        if not (hasattr(row, "x") and hasattr(row, "y")):
            raise ValueError(
                "tiles_csv must contain 'x' and 'y' columns for distance eval")
        return float(getattr(row, "x")), float(getattr(row, "y"))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tiles_csv", required=True,
                    help="CSV with patch_name, split, and coords")
    ap.add_argument("--tiles_root", required=True,
                    help="Folder with QUERY tiles (e.g. test tiles)")

    ap.add_argument("--tile_index", required=True, help="FAISS index path")
    ap.add_argument("--tile_meta", required=True,
                    help="Meta for index (sqlite/csv/parquet)")
    ap.add_argument("--index_tiles_root", required=True,
                    help="Folder with INDEX tiles (train). Used for save_top5")

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--save_top5", action="store_true")
    ap.add_argument("--copy_mode", default="symlink",
                    choices=["copy", "symlink", "hardlink"])

    ap.add_argument("--compute_recall", action="store_true",
                    help="Compute Recall@K only if tiles_csv has column 'faiss_id'")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    index = faiss.read_index(args.tile_index)
    meta = load_meta(args.tile_meta)

    id2coord, coord_kind = build_id2coord(meta)
    id2img = build_id2img(meta, args.index_tiles_root)

    dist_fn = haversine_m if coord_kind == "latlon" else dist_xy

    tiles = pd.read_csv(args.tiles_csv, sep=None, engine="python")
    if "split" not in tiles.columns:
        raise ValueError("tiles_csv must contain column 'split'")
    if "patch_name" not in tiles.columns:
        raise ValueError("tiles_csv must contain column 'patch_name'")

    tiles = tiles[tiles["split"].astype(str) == str(
        args.eval_split)].reset_index(drop=True)
    if tiles.empty:
        raise ValueError(
            f"No rows for eval_split='{args.eval_split}' in {args.tiles_csv}")

    can_recall = bool(args.compute_recall) and ("faiss_id" in tiles.columns)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tfm_query = build_transforms(args.image_size, mode="eval")

    model = DinoV2Encoder(backbone_name=args.backbone,
                          proj_dim=args.proj_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    hits1 = hits5 = hits10 = 0
    dists: List[float] = []
    rows = []

    save_root = os.path.join(args.out_dir, "top5")
    if args.save_top5:
        os.makedirs(save_root, exist_ok=True)

    for local_idx, r in tqdm(list(enumerate(tiles.itertuples(index=False))), desc="Eval"):
        patch_name = str(getattr(r, "patch_name"))
        q_img_path = infer_tile_path(args.tiles_root, patch_name)
        if not os.path.exists(q_img_path):
            raise FileNotFoundError(f"Query tile not found: {q_img_path}")

        tx, ty = get_true_coord_from_csv(r, coord_kind)

        img = open_image_any(q_img_path)
        x = tfm_query(img).unsqueeze(0).to(device)
        z = model(x).detach().cpu().numpy().astype("float32")

        scores, idxs = index.search(z, args.K)
        cand_ids = [int(i) for i in idxs[0].tolist()]
        cand_scores = [float(s) for s in scores[0].tolist()]

        pred_id = cand_ids[0]
        px, py = id2coord.get(pred_id, (float("nan"), float("nan")))
        dm = dist_fn(tx, ty, px, py) if np.isfinite(px) else float("nan")
        dists.append(dm)

        true_faiss_id = int(getattr(r, "faiss_id")) if can_recall else -1
        rank_true = -1
        if can_recall:
            for rrk, fid in enumerate(cand_ids, start=1):
                if fid == true_faiss_id:
                    rank_true = rrk
                    break
            if rank_true != -1:
                if rank_true <= 1:
                    hits1 += 1
                if rank_true <= 5:
                    hits5 += 1
                if rank_true <= 10:
                    hits10 += 1

        if args.save_top5:
            qdir = os.path.join(save_root, safe_name(patch_name))
            os.makedirs(qdir, exist_ok=True)

            q_ext = os.path.splitext(q_img_path)[1].lower()
            q_dst = os.path.join(qdir, f"query{q_ext}")
            if not os.path.exists(q_dst):
                copy_or_symlink(q_img_path, q_dst, mode=args.copy_mode)

            for rrk, (fid, sc) in enumerate(zip(cand_ids[:5], cand_scores[:5]), start=1):
                src = id2img.get(fid, "")
                if not src or not os.path.exists(src):
                    continue
                ext = os.path.splitext(src)[1].lower()
                base = safe_name(os.path.splitext(os.path.basename(src))[0])
                dst = os.path.join(
                    qdir, f"rank_{rrk:02d}__id_{fid}__score_{sc:.6f}__{base}{ext}")
                copy_or_symlink(src, dst, mode=args.copy_mode)

        s1 = float(scores[0][0]) if scores.size else float("nan")
        s2 = float(scores[0][1]) if args.K >= 2 else float("nan")

        rows.append({
            "patch_name": patch_name,
            "query_tile_path": q_img_path,
            "true_faiss_id": true_faiss_id,
            "pred_faiss_id": pred_id,
            "rank_true": rank_true,
            "dist_m_or_units": dm,
            "s1": s1,
            "s2": s2,
            "conf_s1_s2": (s1 - s2) if np.isfinite(s2) else float("nan"),
            "coord_kind": coord_kind,
            "top5_ids": cand_ids[:5],
            "top5_scores": cand_scores[:5],
        })

    n = len(rows)
    dnp = np.array([d for d in dists if np.isfinite(d)], dtype=np.float64)

    summary = {
        "n": int(n),
        "eval_split": str(args.eval_split),
        "coord_kind": coord_kind,
        "median_dist": float(np.median(dnp)) if dnp.size else float("nan"),
        "mean_dist": float(np.mean(dnp)) if dnp.size else float("nan"),
        "p90_dist": float(np.quantile(dnp, 0.90)) if dnp.size else float("nan"),
        "p95_dist": float(np.quantile(dnp, 0.95)) if dnp.size else float("nan"),
    }

    if dnp.size:
        if coord_kind == "latlon":
            summary.update({
                "within_250m": float(np.mean(dnp <= 250.0)),
                "within_500m": float(np.mean(dnp <= 500.0)),
                "within_1000m": float(np.mean(dnp <= 1000.0)),
                "within_2000m": float(np.mean(dnp <= 2000.0)),
            })
        else:
            summary.update({
                "within_250": float(np.mean(dnp <= 250.0)),
                "within_500": float(np.mean(dnp <= 500.0)),
                "within_1000": float(np.mean(dnp <= 1000.0)),
                "within_2000": float(np.mean(dnp <= 2000.0)),
            })

    if can_recall and n > 0:
        summary.update({
            "Recall@1": float(hits1 / n),
            "Recall@5": float(hits5 / n),
            "Recall@10": float(hits10 / n),
        })
    else:
        summary.update({
            "Recall@1": None,
            "Recall@5": None,
            "Recall@10": None,
        })

    pd.DataFrame(rows).to_csv(os.path.join(
        args.out_dir, "per_query.csv"), index=False)
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
