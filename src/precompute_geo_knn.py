import argparse
import json
import os
import sqlite3
from typing import Tuple

import faiss
import numpy as np
import pandas as pd


def resolve_existing_file(path: str, arg_name: str) -> str:
    p = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{arg_name} not found: '{path}' -> '{p}'. "
            f"Current working dir: '{os.getcwd()}'"
        )
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{arg_name} is not a file: '{p}'")
    if not os.access(p, os.R_OK):
        raise PermissionError(f"{arg_name} is not readable: '{p}'")
    return p


def latlon_to_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    lat0r = np.deg2rad(lat0)
    x = (lon - lon0) * np.cos(lat0r) * 111320.0
    y = (lat - lat0) * 110540.0
    return x, y


def load_sqlite_meta(meta_path: str) -> pd.DataFrame:
    try:
        con = sqlite3.connect(meta_path)
    except sqlite3.Error as e:
        raise RuntimeError(
            f"Failed to open anchors_meta sqlite: '{meta_path}'. "
            f"sqlite3 error: {e}"
        ) from e

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
                "anchors_meta sqlite must contain table 'anchor_meta' or 'tile_meta'")
    except Exception as e:
        raise RuntimeError(
            f"Failed reading anchors_meta sqlite: '{meta_path}'. "
            f"Error: {e}"
        ) from e
    finally:
        con.close()

    if id_col not in df.columns:
        raise ValueError(f"anchors_meta missing id column '{id_col}'")

    df = df.copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    if df[id_col].isna().any():
        raise ValueError("anchors_meta has non-numeric ids")

    df[id_col] = df[id_col].astype(np.int64)
    df = df.sort_values(id_col).reset_index(drop=True)

    ids = df[id_col].to_numpy(dtype=np.int64)
    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError(f"{id_col} must be contiguous 0..N-1 without gaps")

    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute geo-kNN anchor candidates per train row")
    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sep", default="auto")
    ap.add_argument("--anchors_meta", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--use_xy", action="store_true")
    args = ap.parse_args()

    if args.K <= 0:
        raise ValueError("--K must be > 0")

    anchors_meta_path = resolve_existing_file(
        args.anchors_meta, "--anchors_meta")
    tiles_csv_path = resolve_existing_file(args.tiles_csv, "--tiles_csv")

    meta = load_sqlite_meta(anchors_meta_path)
    n_anchors = len(meta)
    if args.K > n_anchors:
        raise ValueError(
            f"K={args.K} is larger than number of anchors N={n_anchors}")

    has_anchor_xy = {"x", "y"}.issubset(meta.columns)
    has_anchor_latlon = {"lat", "lon"}.issubset(meta.columns)

    if args.use_xy and has_anchor_xy:
        anchor_x = meta["x"].astype(np.float64).to_numpy()
        anchor_y = meta["y"].astype(np.float64).to_numpy()
        anchor_xy = np.stack([anchor_x, anchor_y], axis=1).astype(np.float32)

        if has_anchor_latlon:
            lat0 = float(meta["lat"].astype(np.float64).mean())
            lon0 = float(meta["lon"].astype(np.float64).mean())
        else:
            lat0 = None
            lon0 = None
    else:
        if not has_anchor_latlon:
            raise ValueError(
                "anchors_meta must contain lat/lon (or x/y with --use_xy)")
        a_lat = meta["lat"].astype(np.float64).to_numpy()
        a_lon = meta["lon"].astype(np.float64).to_numpy()
        lat0 = float(a_lat.mean())
        lon0 = float(a_lon.mean())
        ax, ay = latlon_to_xy_m(a_lat, a_lon, lat0, lon0)
        anchor_xy = np.stack([ax, ay], axis=1).astype(np.float32)

    if args.sep == "auto":
        tiles = pd.read_csv(tiles_csv_path, sep=None, engine="python")
    else:
        sep_real = "\t" if args.sep == "\\t" else args.sep
        tiles = pd.read_csv(tiles_csv_path, sep=sep_real)

    for c in ("patch_name", "split"):
        if c not in tiles.columns:
            raise ValueError(f"tiles_csv must contain '{c}'")

    tiles = tiles[tiles["split"].astype(str) == str(
        args.split)].reset_index(drop=True)
    if tiles.empty:
        raise ValueError(f"No rows for split='{args.split}'")

    has_query_xy = {"x", "y"}.issubset(tiles.columns)
    has_query_latlon = {"lat", "lon"}.issubset(tiles.columns)

    if args.use_xy and has_query_xy and has_anchor_xy:
        qx = tiles["x"].astype(np.float64).to_numpy()
        qy = tiles["y"].astype(np.float64).to_numpy()
        query_xy = np.stack([qx, qy], axis=1).astype(np.float32)
    else:
        if not has_query_latlon:
            raise ValueError(
                "tiles_csv must contain lat/lon for geo-kNN precompute")
        if lat0 is None or lon0 is None:
            raise ValueError(
                "Cannot project query lat/lon because anchor lat0/lon0 is unavailable")

        q_lat = tiles["lat"].astype(np.float64).to_numpy()
        q_lon = tiles["lon"].astype(np.float64).to_numpy()
        qx, qy = latlon_to_xy_m(q_lat, q_lon, lat0, lon0)
        query_xy = np.stack([qx, qy], axis=1).astype(np.float32)

    index = faiss.IndexFlatL2(2)
    index.add(anchor_xy)
    _, idx = index.search(query_xy, int(args.K))

    idx = idx.astype(np.int64, copy=False)

    out_dir = os.path.dirname(os.path.abspath(args.out_npy))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.save(args.out_npy, idx)

    info = {
        "K": int(args.K),
        "lat0": lat0,
        "lon0": lon0,
        "anchors_meta": anchors_meta_path,
        "tiles_csv": tiles_csv_path,
        "split": str(args.split),
        "N_train": int(idx.shape[0]),
        "N_anchors": int(n_anchors),
        "use_xy": bool(args.use_xy),
    }

    info_path = args.out_npy + ".json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
