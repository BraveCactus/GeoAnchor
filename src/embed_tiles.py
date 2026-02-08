from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def save_meta(df: pd.DataFrame, out_meta: Path) -> None:
    suffix = out_meta.suffix.lower()
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    if suffix in {".sqlite", ".db"}:
        with sqlite3.connect(out_meta) as conn:
            df.to_sql("tile_meta", conn, if_exists="replace", index=False)
    elif suffix == ".parquet":
        try:
            df.to_parquet(out_meta, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Не удалось записать parquet. Установите pyarrow/fastparquet или используйте --out-meta *.sqlite"
            ) from exc
    elif suffix == ".csv":
        df.to_csv(out_meta, index=False)
    else:
        raise ValueError(
            "Неподдерживаемый формат --out-meta. Используйте .sqlite, .parquet или .csv")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from CSV with precomputed embeddings"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV with columns: patch_name,original_image,tile_path,lat,lon,split,embedding...",
    )
    parser.add_argument("--out-index", type=Path,
                        default=Path("app/data/city.index.faiss"))
    parser.add_argument("--out-meta", type=Path,
                        default=Path("app/data/tile_meta.sqlite"))
    parser.add_argument("--out-vectors", type=Path,
                        default=Path("app/data/city.vectors.npy"))
    parser.add_argument("--normalize", action="store_true",
                        help="Apply L2 normalization before indexing")
    parser.add_argument(
        "--metric",
        choices=["ip", "l2"],
        default="ip",
        help="FAISS metric; use ip for cosine similarity if vectors are L2-normalized",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="If CSV has 'split', filter by this value (e.g., train/val/test)",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=",",
        help="CSV separator (default ','). Use '\\t' for TSV or ';' for semicolon CSV.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable pandas low_memory mode (may infer dtypes in chunks).",
    )
    args = parser.parse_args()

    sep = "\t" if args.sep == "\\t" else args.sep
    df = pd.read_csv(args.csv, sep=sep, low_memory=args.low_memory)

    required = {"patch_name", "original_image",
                "tile_path", "lat", "lon", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if args.split:
        df = df[df["split"].astype(str) == args.split].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"После фильтра split='{args.split}' не осталось строк.")

    if "embedding" not in df.columns:
        raise ValueError(
            "CSV должен содержать колонку 'embedding' (с неё начинаются числа эмбеддинга)")

    emb_start_idx = df.columns.get_loc("embedding")
    emb_cols = list(df.columns[emb_start_idx:])

    emb_df = df.loc[:, emb_cols].apply(pd.to_numeric, errors="coerce")

    if emb_df.isna().any().any():
        bad_cols = emb_df.columns[emb_df.isna().any()].tolist()[:10]
        bad_rows = emb_df[emb_df.isna().any(axis=1)].index.to_list()[:5]
        raise ValueError(
            "В эмбеддингах найдены нечисловые/пустые значения (NaN) после приведения к float.\n"
            f"Примеры колонок с NaN: {bad_cols}\n"
            f"Примеры строк с NaN: {bad_rows}\n"
            "Проверь, что CSV не обрезан, и что разделитель --sep задан правильно."
        )

    vectors = emb_df.to_numpy(dtype=np.float32, copy=False)

    if vectors.ndim != 2 or vectors.shape[0] != len(df):
        raise ValueError(f"Bad vectors shape: {vectors.shape}")

    if args.normalize:
        vectors = l2_normalize(vectors).astype("float32", copy=False)

    d = vectors.shape[1]
    if args.metric == "ip":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(vectors)

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    args.out_vectors.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(args.out_index))
    np.save(args.out_vectors, vectors.astype("float32", copy=False))

    meta_cols = ["patch_name", "original_image",
                 "tile_path", "lat", "lon", "split"]
    meta = df[meta_cols].copy().reset_index(drop=True)
    meta["faiss_id"] = meta.index.astype(int)

    save_meta(meta, args.out_meta)

    print(f"Saved index   -> {args.out_index}")
    print(f"Saved vectors -> {args.out_vectors}  shape={vectors.shape}")
    print(f"Saved meta    -> {args.out_meta}  rows={len(meta)}")
    print(f"Metric={args.metric}, normalize={args.normalize}, dim={d}")


if __name__ == "__main__":
    main()
