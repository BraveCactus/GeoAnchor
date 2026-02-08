# src/infer_folder.py
import os
import json
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import faiss
import sqlite3
import tifffile as tiff

from data.augs import build_transforms
from model.dinov2_gem import DinoV2Encoder


IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp")


def open_image_any(path: str) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tiff.imread(path)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            # CHW -> HWC if needed
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


def infer_tile_path(tiles_root: str, patch_name: str,
                    exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")) -> str:
    base = os.path.join(tiles_root, patch_name)
    for ext in exts:
        p = base + ext
        if os.path.exists(p):
            return p
    return base + exts[0]


def list_images(root: str, recursive: bool = True):
    root = os.path.abspath(root)
    out = []
    if recursive:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(IMG_EXTS):
                    out.append(os.path.join(dp, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and fn.lower().endswith(IMG_EXTS):
                out.append(p)
    out.sort()
    return out


def safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    for c in bad:
        s = s.replace(c, "_")
    s = s.strip().strip(".")
    return s or "query"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_file(src: str, dst: str, mode: str):
    ensure_dir(os.path.dirname(dst))
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if os.path.exists(dst):
            return
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        if os.path.exists(dst):
            return
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown copy_mode: {mode}")


def resolve_under_root(p: str, root: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    root = str(root).strip().strip('"').strip("'")
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--queries_dir", required=True,
                    help="Folder with query images")
    ap.add_argument("--recursive", action="store_true",
                    help="Recursively scan queries_dir")

    ap.add_argument("--tile_index", required=True, help="FAISS index path")
    ap.add_argument("--tile_meta", required=True,
                    help="Meta path (.sqlite/.db/.csv/.parquet)")

    ap.add_argument("--train_root", required=True,
                    help="Root folder with TRAIN tile images (can be relative). Used to resolve meta relative paths or patch_name.")
    ap.add_argument("--meta_path_col", default="",
                    help="Optional: explicitly specify meta column with image path")

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--K", type=int, default=10)

    ap.add_argument("--save_topk_images", action="store_true",
                    help="Save Top-K retrieved images into out_dir/<query_name>/")
    ap.add_argument("--copy_mode", default="copy", choices=["copy", "symlink", "hardlink"],
                    help="How to save result images (default: copy)")
    ap.add_argument("--save_query", action="store_true",
                    help="Also save the query image into its folder")

    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    index = faiss.read_index(args.tile_index)

    meta = load_meta(args.tile_meta)
    if "faiss_id" not in meta.columns:
        raise ValueError("tile_meta must contain 'faiss_id' column")

    candidate_cols = ["path", "tile_path", "img_path",
                      "image_path", "filepath", "file_path"]
    path_col = args.meta_path_col.strip() if args.meta_path_col.strip() else ""

    if path_col:
        if path_col not in meta.columns:
            raise ValueError(f"--meta_path_col='{path_col}' not found in meta")
    else:
        for c in candidate_cols:
            if c in meta.columns:
                path_col = c
                break

    has_patch_name = "patch_name" in meta.columns

    if not path_col and not has_patch_name:
        raise ValueError(
            "tile_meta must contain either a path column (path/tile_path/img_path/...) "
            "or patch_name."
        )

    train_root = os.path.abspath(args.train_root)

    id2path = {}
    for r in meta.itertuples(index=False):
        fid = int(getattr(r, "faiss_id"))
        if path_col:
            raw = str(getattr(r, path_col))
            id2path[fid] = resolve_under_root(raw, train_root)
        else:
            patch = str(getattr(r, "patch_name"))
            id2path[fid] = infer_tile_path(train_root, patch)

    sample = list(id2path.values())[:20]
    if sample:
        missing = sum(0 if os.path.exists(p) else 1 for p in sample)
        print(f"[check] sample tile paths missing: {missing}/{len(sample)}")
        if missing == len(sample):
            print("[check] WARNING: none of sampled tile paths exist. "
                  "train_root/meta_path_col/patch_name mapping is likely wrong.")

    queries = list_images(args.queries_dir, recursive=args.recursive)
    if not queries:
        raise ValueError(f"No images found in {args.queries_dir}")

    tfm_query = build_transforms(args.image_size, mode="eval")
    model = DinoV2Encoder(backbone_name=args.backbone,
                          proj_dim=args.proj_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows = []
    jsonl_path = os.path.join(args.out_dir, "results.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for qpath in tqdm(queries, desc="InferFolder"):
            qstem = safe_name(os.path.splitext(os.path.basename(qpath))[0])
            qsubdir = os.path.join(args.out_dir, qstem)
            if args.save_topk_images:
                ensure_dir(qsubdir)

            if args.save_topk_images and args.save_query:
                qdst = os.path.join(qsubdir, "query" +
                                    os.path.splitext(qpath)[1].lower())
                if not os.path.exists(qdst):
                    save_file(qpath, qdst, args.copy_mode)

            img = open_image_any(qpath)
            x = tfm_query(img).unsqueeze(0).to(device)
            z = model(x).cpu().numpy().astype("float32")

            scores, idxs = index.search(z, args.K)
            cand_ids = [int(i) for i in idxs[0].tolist()]
            cand_scores = [float(s) for s in scores[0].tolist()]

            saved = []
            for rank, (fid, sc) in enumerate(zip(cand_ids, cand_scores), start=1):
                src = id2path.get(fid, "")
                ok = bool(src) and os.path.exists(src)

                saved_path = ""
                if args.save_topk_images and ok:
                    ext = os.path.splitext(src)[1].lower()
                    base = safe_name(os.path.splitext(
                        os.path.basename(src))[0])
                    fname = f"rank_{rank:02d}__id_{fid}__score_{sc:.6f}__{base}{ext}"
                    saved_path = os.path.join(qsubdir, fname)
                    save_file(src, saved_path, args.copy_mode)

                saved.append({
                    "rank": rank,
                    "faiss_id": fid,
                    "score": sc,
                    "tile_src": src,
                    "saved_path": saved_path,
                    "exists": ok,
                })

                rows.append({
                    "query_path": qpath,
                    "query_name": qstem,
                    "rank": rank,
                    "faiss_id": fid,
                    "score": sc,
                    "tile_src": src,
                    "saved_path": saved_path,
                    "exists": ok,
                })

            rec = {"query_path": qpath, "query_name": qstem, "topk": saved}
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_csv(os.path.join(
        args.out_dir, "results.csv"), index=False)
    print(
        f"Done. Wrote:\n- {os.path.join(args.out_dir, 'results.csv')}\n- {jsonl_path}")
    if args.save_topk_images:
        print(f"- per-query folders in {args.out_dir}")


if __name__ == "__main__":
    main()
