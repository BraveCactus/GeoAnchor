from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
import tifffile as tiff
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.augs import build_transforms
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


def _basename_no_windows_dirs(name: str) -> str:
    return os.path.basename(str(name).replace("\\", "/"))


def infer_path_by_name(root: str, name: str, exts: Tuple[str, ...] = IMAGE_EXTS) -> str:
    clean_name = _basename_no_windows_dirs(name)
    base = os.path.join(root, clean_name)

    if os.path.splitext(clean_name)[1].lower() in exts and os.path.exists(base):
        return base

    for ext in exts:
        p = base + ext
        if os.path.exists(p):
            return p

    return base + exts[0]


def safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = str(s)
    for c in bad:
        out = out.replace(c, "_")
    out = out.strip().strip(".")
    return out or "query"


def copy_or_symlink(src: str, dst: str, mode: str) -> None:
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
        raise ValueError(f"Unknown copy_mode: {mode}")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * \
        math.cos(p2) * math.sin(dl / 2.0) ** 2
    return float(2.0 * r * math.asin(math.sqrt(a)))


def load_meta(meta_path: str) -> pd.DataFrame:
    p = meta_path.lower()
    if p.endswith(".sqlite") or p.endswith(".db"):
        with sqlite3.connect(meta_path) as conn:
            return pd.read_sql("SELECT * FROM tile_meta", conn)
    if p.endswith(".csv"):
        return pd.read_csv(meta_path)
    if p.endswith(".parquet"):
        return pd.read_parquet(meta_path)
    raise ValueError(
        "Unsupported tile_meta format. Use .sqlite/.db/.csv/.parquet")


def resolve_index_image_path(row: pd.Series, index_tiles_root: str, exts: Tuple[str, ...] = IMAGE_EXTS) -> str:
    tile_name = ""
    for c in ("tile_name", "patch_name"):
        if c in row and pd.notna(row[c]):
            tile_name = str(row[c])
            break

    raw_path = ""
    if "tile_path" in row and pd.notna(row["tile_path"]):
        raw_path = str(row["tile_path"]).strip().strip('"').strip("'")

    candidates: List[str] = []

    if raw_path:
        raw_path = raw_path.replace("\\", "/")
        basename = os.path.basename(raw_path)
        if os.path.isabs(raw_path):
            candidates.append(raw_path)
            if basename:
                candidates.append(os.path.join(index_tiles_root, basename))
        else:
            candidates.append(os.path.join(index_tiles_root, raw_path))
            if basename:
                candidates.append(os.path.join(index_tiles_root, basename))

    for c in candidates:
        if os.path.exists(c):
            return c

    if tile_name:
        p = infer_path_by_name(index_tiles_root, tile_name, exts=exts)
        if os.path.exists(p):
            return p

    if candidates:
        return candidates[-1]
    if tile_name:
        return infer_path_by_name(index_tiles_root, tile_name, exts=exts)

    return ""


def validate_index_and_meta(index: faiss.Index, meta: pd.DataFrame, index_tiles_root: str) -> pd.DataFrame:
    required_cols = {"tile_name", "split", "lat", "lon", "faiss_id"}
    missing_cols = [c for c in required_cols if c not in meta.columns]
    if missing_cols:
        raise ValueError(
            f"tile_meta is missing required columns: {missing_cols}")

    ntotal = int(index.ntotal)
    d = int(index.d)
    if ntotal <= 0 or d <= 0:
        raise ValueError(f"Invalid FAISS index: ntotal={ntotal}, d={d}")

    if meta.empty:
        raise ValueError("tile_meta is empty")

    work = meta.copy()
    work["faiss_id"] = pd.to_numeric(work["faiss_id"], errors="coerce")
    if work["faiss_id"].isna().any():
        bad = int(work["faiss_id"].isna().sum())
        raise ValueError(f"tile_meta has non-numeric faiss_id in {bad} rows")

    work["faiss_id"] = work["faiss_id"].astype(np.int64)

    if work["faiss_id"].nunique() != ntotal:
        raise ValueError(
            f"tile_meta unique faiss_id count mismatch: unique={work['faiss_id'].nunique()} vs index.ntotal={ntotal}"
        )

    expected = np.arange(ntotal, dtype=np.int64)
    actual = np.sort(work["faiss_id"].to_numpy())
    if not np.array_equal(actual, expected):
        missing = sorted(set(expected.tolist()) - set(actual.tolist()))
        extra = sorted(set(actual.tolist()) - set(expected.tolist()))
        raise ValueError(
            f"faiss_id must be contiguous 0..{ntotal-1} without holes; missing={missing[:10]}, extra={extra[:10]}"
        )

    work = work.sort_values("faiss_id").reset_index(drop=True)

    sample_n = min(20, ntotal)
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(ntotal, size=sample_n, replace=False)
    errs: List[str] = []

    for fid in sample_ids.tolist():
        row = work.iloc[fid]
        tile_name = str(row["tile_name"]) if pd.notna(row["tile_name"]) else ""
        if not tile_name.strip():
            errs.append(f"faiss_id={fid}: empty tile_name")

        lat = float(row["lat"]) if pd.notna(row["lat"]) else float("nan")
        lon = float(row["lon"]) if pd.notna(row["lon"]) else float("nan")
        if (not np.isfinite(lat)) or (not np.isfinite(lon)):
            errs.append(
                f"faiss_id={fid}: non-finite lat/lon lat={lat} lon={lon}")

        p = resolve_index_image_path(row, index_tiles_root)
        if not p or not os.path.exists(p):
            errs.append(
                f"faiss_id={fid}: index image not found (resolved='{p}')")

    if errs:
        raise RuntimeError("Sanity checks failed:\n" + "\n".join(errs))

    return work


class QueryDataset(Dataset):
    def __init__(self, tiles_csv: str, eval_split: str, tiles_root: str, image_size: int, sep: str = "auto"):
        if sep == "auto":
            df = pd.read_csv(tiles_csv, sep=None, engine="python")
        else:
            sep_real = "\t" if sep == "\\t" else sep
            df = pd.read_csv(tiles_csv, sep=sep_real)

        for c in ("patch_name", "split", "lat", "lon"):
            if c not in df.columns:
                raise ValueError(f"tiles_csv must contain '{c}' column")

        df = df[df["split"].astype(str) == str(
            eval_split)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows for split='{eval_split}'")

        self.df = df
        self.tiles_root = os.path.abspath(tiles_root)
        self.tfm = build_transforms(image_size, mode="eval")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patch_name = str(row["patch_name"])
        img_path = infer_path_by_name(self.tiles_root, patch_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Query image not found: {img_path} (patch_name={patch_name})")

        img = open_image_any(img_path)
        x = self.tfm(img)

        q_lat = float(row["lat"])
        q_lon = float(row["lon"])
        if (not np.isfinite(q_lat)) or (not np.isfinite(q_lon)):
            raise ValueError(
                f"Non-finite query lat/lon for patch_name={patch_name}: lat={q_lat} lon={q_lon}")

        return x, patch_name, img_path, q_lat, q_lon


def load_encoder(ckpt_path: str, backbone: str, proj_dim: int, device: torch.device) -> DinoV2Encoder:
    model = DinoV2Encoder(backbone_name=backbone, proj_dim=proj_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if (isinstance(ckpt, dict)
                           and "model" in ckpt) else ckpt

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

    if len(missing) > 0:
        print(f"[warn] Missing keys while loading encoder: {len(missing)}")

    model.eval()
    return model


def metric_consistency_check(index: faiss.Index, metric: str) -> None:
    if not hasattr(index, "metric_type"):
        return

    mt = int(index.metric_type)
    if metric == "ip" and mt != int(faiss.METRIC_INNER_PRODUCT):
        raise ValueError(
            "--metric ip requested, but FAISS index metric_type is not inner product")
    if metric == "l2" and mt != int(faiss.METRIC_L2):
        raise ValueError(
            "--metric l2 requested, but FAISS index metric_type is not L2")


def summarize_distances(dist_m: np.ndarray) -> Dict[str, float]:
    if dist_m.size == 0:
        return {
            "median_m": float("nan"),
            "mean_m": float("nan"),
            "p90_m": float("nan"),
            "p95_m": float("nan"),
            "within_250m": float("nan"),
            "within_500m": float("nan"),
            "within_1000m": float("nan"),
            "within_2000m": float("nan"),
        }

    return {
        "median_m": float(np.median(dist_m)),
        "mean_m": float(np.mean(dist_m)),
        "p90_m": float(np.quantile(dist_m, 0.90)),
        "p95_m": float(np.quantile(dist_m, 0.95)),
        "within_250m": float((dist_m <= 250.0).mean()),
        "within_500m": float((dist_m <= 500.0).mean()),
        "within_1000m": float((dist_m <= 1000.0).mean()),
        "within_2000m": float((dist_m <= 2000.0).mean()),
    }


def latlon_to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(lat.astype(np.float64))
    lon_r = np.deg2rad(lon.astype(np.float64))
    cos_lat = np.cos(lat_r)
    x = cos_lat * np.cos(lon_r)
    y = cos_lat * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def compute_oracle_nn_haversine(
    q_lat: np.ndarray,
    q_lon: np.ndarray,
    bank_lat: np.ndarray,
    bank_lon: np.ndarray,
) -> np.ndarray:
    bank_xyz = latlon_to_unit_xyz(bank_lat, bank_lon)
    q_xyz = latlon_to_unit_xyz(q_lat, q_lon)

    idx = faiss.IndexFlatIP(3)
    idx.add(bank_xyz)
    _, nn = idx.search(q_xyz, 1)
    nn = nn[:, 0]

    out = np.empty(q_lat.shape[0], dtype=np.float64)
    for i in range(q_lat.shape[0]):
        j = int(nn[i])
        out[i] = haversine_m(q_lat[i], q_lon[i], bank_lat[j], bank_lon[j])
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Robust retrieval eval for GEO")

    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--tiles_root", required=True)

    ap.add_argument("--tile_index", required=True)
    ap.add_argument("--tile_meta", required=True)
    ap.add_argument("--index_tiles_root", default="",
                    help="Root of indexed train images (used for sanity checks and save_topk).")

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--metric", choices=["ip", "l2"], default="ip")
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sep", default="auto")

    ap.add_argument("--compute_oracle_nn", action="store_true")

    ap.add_argument("--save_topk", type=int, default=0)
    ap.add_argument(
        "--copy_mode", choices=["copy", "symlink", "hardlink"], default="copy")

    ap.add_argument("--compute_recall", action="store_true",
                    help="Optional recall@K if tiles_csv has valid faiss_id.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    index_tiles_root = args.index_tiles_root.strip() or args.tiles_root
    index_tiles_root = os.path.abspath(index_tiles_root)

    index = faiss.read_index(args.tile_index)
    metric_consistency_check(index, args.metric)

    meta_raw = load_meta(args.tile_meta)
    meta = validate_index_and_meta(index, meta_raw, index_tiles_root)

    ntotal = int(index.ntotal)
    d = int(index.d)
    print(f"[info] Loaded index: ntotal={ntotal}, d={d}")

    id2lat = meta["lat"].astype(float).to_numpy(dtype=np.float64)
    id2lon = meta["lon"].astype(float).to_numpy(dtype=np.float64)

    ds = QueryDataset(
        tiles_csv=args.tiles_csv,
        eval_split=args.eval_split,
        tiles_root=args.tiles_root,
        image_size=args.image_size,
        sep=args.sep,
    )
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True)

    if args.sep == "auto":
        tiles_full = pd.read_csv(args.tiles_csv, sep=None, engine="python")
    else:
        sep_real = "\t" if args.sep == "\\t" else args.sep
        tiles_full = pd.read_csv(args.tiles_csv, sep=sep_real)
    tiles_eval = tiles_full[tiles_full["split"].astype(
        str) == str(args.eval_split)].reset_index(drop=True)
    can_recall = bool(args.compute_recall) and (
        "faiss_id" in tiles_eval.columns)

    if can_recall:
        fid_vals = pd.to_numeric(tiles_eval["faiss_id"], errors="coerce")
        valid_recall_rows = fid_vals.notna() & (fid_vals >= 0) & (fid_vals < ntotal)
        can_recall = bool(valid_recall_rows.all())
        if not can_recall:
            print("[warn] --compute_recall requested, but faiss_id in tiles_csv is not fully valid for eval_split; recall disabled")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_encoder(args.ckpt, args.backbone, args.proj_dim, device)

    if d != int(args.proj_dim):
        raise ValueError(
            f"Index dimension mismatch: index.d={d}, proj_dim={args.proj_dim}")

    rows: List[Dict[str, object]] = []
    dist_vals: List[float] = []

    hits1 = 0
    hits5 = 0
    hits10 = 0

    topk_root = os.path.join(args.out_dir, "topk")
    if args.save_topk > 0:
        os.makedirs(topk_root, exist_ok=True)

    offset = 0
    search_k = min(int(args.K), ntotal)
    if search_k <= 0:
        raise ValueError("--K must be >= 1")

    for x, patch_names, query_paths, q_lat, q_lon in tqdm(dl, desc="Evaluating"):
        x = x.to(device, non_blocking=True)
        q_emb = model(x).detach().cpu().numpy().astype(np.float32)

        if args.normalize:
            faiss.normalize_L2(q_emb)

        scores, ids = index.search(q_emb, search_k)

        bsz = q_emb.shape[0]
        for bi in range(bsz):
            patch_name = str(patch_names[bi])
            query_path = str(query_paths[bi])
            lat = float(q_lat[bi].item())
            lon = float(q_lon[bi].item())

            top_ids = [int(v) for v in ids[bi].tolist()]
            top_scores = [float(v) for v in scores[bi].tolist()]

            pred_id = int(top_ids[0])
            pred_lat = float(id2lat[pred_id])
            pred_lon = float(id2lon[pred_id])
            dist_m = haversine_m(lat, lon, pred_lat, pred_lon)
            dist_vals.append(dist_m)

            row: Dict[str, object] = {
                "patch_name": patch_name,
                "tile_path_query": query_path,
                "q_lat": lat,
                "q_lon": lon,
                "pred_faiss_id": pred_id,
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "dist_m": dist_m,
                "topk_ids": json.dumps(top_ids, ensure_ascii=False),
                "topk_scores": json.dumps(top_scores, ensure_ascii=False),
            }

            if can_recall:
                true_id = int(tiles_eval.iloc[offset + bi]["faiss_id"])
                rank_true = -1
                for rnk, fid in enumerate(top_ids, start=1):
                    if fid == true_id:
                        rank_true = rnk
                        break
                if rank_true != -1:
                    if rank_true <= 1:
                        hits1 += 1
                    if rank_true <= 5:
                        hits5 += 1
                    if rank_true <= 10:
                        hits10 += 1
                row["true_faiss_id"] = true_id
                row["rank_true"] = rank_true

            rows.append(row)

            if args.save_topk > 0:
                qdir = os.path.join(topk_root, safe_name(patch_name))
                os.makedirs(qdir, exist_ok=True)

                q_ext = os.path.splitext(query_path)[1].lower() or ".png"
                q_dst = os.path.join(qdir, f"query{q_ext}")
                if not os.path.exists(q_dst):
                    copy_or_symlink(query_path, q_dst, args.copy_mode)

                max_k = min(args.save_topk, len(top_ids))
                for rk in range(max_k):
                    fid = int(top_ids[rk])
                    score = float(top_scores[rk])
                    bank_row = meta.iloc[fid]
                    src = resolve_index_image_path(bank_row, index_tiles_root)
                    if not src or not os.path.exists(src):
                        continue
                    ext = os.path.splitext(src)[1].lower() or ".png"
                    dst = os.path.join(
                        qdir,
                        f"rank_{rk+1:02d}__faiss_{fid}__score_{score:.6f}{ext}",
                    )
                    copy_or_symlink(src, dst, args.copy_mode)

        offset += bsz

    dist_arr = np.array(dist_vals, dtype=np.float64)
    summary = {
        "n": int(len(rows)),
        "coord_kind": "latlon",
        "eval_split": str(args.eval_split),
        "index_ntotal": ntotal,
        "index_dim": d,
        **summarize_distances(dist_arr),
    }

    if can_recall and len(rows) > 0:
        n = float(len(rows))
        summary["Recall@1"] = float(hits1 / n)
        summary["Recall@5"] = float(hits5 / n)
        summary["Recall@10"] = float(hits10 / n)

    per_query_path = os.path.join(args.out_dir, "per_query.csv")
    pd.DataFrame(rows).to_csv(per_query_path, index=False)

    summary_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.compute_oracle_nn:
        q_df = pd.DataFrame(rows)
        q_lat_np = q_df["q_lat"].to_numpy(dtype=np.float64)
        q_lon_np = q_df["q_lon"].to_numpy(dtype=np.float64)

        oracle_dist = compute_oracle_nn_haversine(
            q_lat=q_lat_np,
            q_lon=q_lon_np,
            bank_lat=id2lat,
            bank_lon=id2lon,
        )

        oracle_summary = {
            "n": int(oracle_dist.shape[0]),
            "coord_kind": "latlon",
            "oracle_median_m": float(np.median(oracle_dist)),
            "oracle_mean_m": float(np.mean(oracle_dist)),
            "oracle_p90_m": float(np.quantile(oracle_dist, 0.90)),
            "oracle_p95_m": float(np.quantile(oracle_dist, 0.95)),
            "oracle_within_250m": float((oracle_dist <= 250.0).mean()),
            "oracle_within_500m": float((oracle_dist <= 500.0).mean()),
            "oracle_within_1000m": float((oracle_dist <= 1000.0).mean()),
            "oracle_within_2000m": float((oracle_dist <= 2000.0).mean()),
        }

        oracle_path = os.path.join(args.out_dir, "oracle_summary.json")
        with open(oracle_path, "w", encoding="utf-8") as f:
            json.dump(oracle_summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(oracle_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
