
import argparse
import json
import math
import os
import random
import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tifffile as tiff
from torch.utils.data import DataLoader, Dataset
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


def latlon_to_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    lat0r = np.deg2rad(lat0)
    x = (lon - lon0) * np.cos(lat0r) * 111320.0
    y = (lat - lat0) * 110540.0
    return x, y


def xy_to_latlon_m(x: torch.Tensor, y: torch.Tensor, lat0: float, lon0: float) -> Tuple[torch.Tensor, torch.Tensor]:
    lat = y / 110540.0 + lat0
    lon = x / (111320.0 * math.cos(math.radians(lat0)) + 1e-12) + lon0
    return lat, lon


def haversine_m_torch(lat1: torch.Tensor, lon1: torch.Tensor, lat2: torch.Tensor, lon2: torch.Tensor) -> torch.Tensor:
    r = 6371000.0
    p1 = torch.deg2rad(lat1)
    p2 = torch.deg2rad(lat2)
    dphi = torch.deg2rad(lat2 - lat1)
    dl = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dphi / 2.0).pow(2) + torch.cos(p1) * \
        torch.cos(p2) * torch.sin(dl / 2.0).pow(2)
    c = 2.0 * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))
    return r * c


def inv_softplus(x: float) -> float:
    # inverse of softplus for initialization
    if x > 20:
        return x
    return float(math.log(math.expm1(max(x, 1e-8))))


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
                "anchors_meta SQLite must contain table 'anchor_meta' or 'tile_meta'")
    finally:
        con.close()

    for c in (id_col, "patch_name", "lat", "lon"):
        if c not in df.columns:
            raise ValueError(f"anchors_meta missing required column '{c}'")

    df = df.copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    if df[id_col].isna().any():
        raise ValueError("anchors_meta has non-numeric id values")

    df[id_col] = df[id_col].astype(np.int64)
    df = df.sort_values(id_col).reset_index(drop=True)

    ids = df[id_col].to_numpy(dtype=np.int64)
    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError(f"{id_col} in anchors_meta must be contiguous 0..N-1")

    lat = df["lat"].astype(np.float64).to_numpy()
    lon = df["lon"].astype(np.float64).to_numpy()
    patch_name = df["patch_name"].astype(str).to_numpy()

    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("anchors_meta contains non-finite lat/lon")

    return lat, lon, patch_name


def load_stageA_into_encoder(enc: DinoV2Encoder, ckpt_init: str):
    ckpt = torch.load(ckpt_init, map_location="cpu")
    sd = ckpt["model"] if (isinstance(ckpt, dict)
                           and "model" in ckpt) else ckpt

    if not isinstance(sd, dict):
        raise ValueError(
            "ckpt_init must be a state_dict or dict with key 'model'")

    if any(k.startswith("enc.") for k in sd.keys()):
        sd = {k[len("enc."):]: v for k, v in sd.items()
              if k.startswith("enc.")}
    elif any(k.startswith("model.enc.") for k in sd.keys()):
        sd = {k[len("model.enc."):]: v for k, v in sd.items()
              if k.startswith("model.enc.")}

    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys while loading ckpt_init into encoder: {unexpected[:5]}")
    if missing:
        print(
            f"[warn] Missing keys while loading ckpt_init into encoder: {len(missing)}")


class AnchorTrainDataset(Dataset):
    """Uses only patch_name + tiles_root. Never uses CSV path columns."""

    def __init__(
        self,
        tiles_csv: str,
        split: str,
        tiles_root: str,
        image_size: int,
        sep: str = "auto",
        geo_knn_npy_path: str = "",
        geo_knn_k: int = 0,
        n_anchors: Optional[int] = None,
    ):
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

        self.geo_knn: Optional[np.ndarray] = None
        if geo_knn_npy_path:
            knn = np.load(geo_knn_npy_path)
            if knn.ndim != 2:
                raise ValueError(
                    f"train_geo_knn_npy must be 2D [N,K], got shape={knn.shape}")
            if knn.shape[0] != len(self.df):
                raise ValueError(
                    "train_geo_knn_npy row count mismatch: "
                    f"npy_rows={knn.shape[0]} vs train_rows={len(self.df)}. "
                    "Пересчитай npy на тот же split и тот же порядок CSV."
                )
            knn = knn.astype(np.int64, copy=False)

            if geo_knn_k > 0:
                if geo_knn_k > knn.shape[1]:
                    raise ValueError(
                        f"geo_knn_K={geo_knn_k} is larger than npy K={knn.shape[1]}"
                    )
                knn = knn[:, :geo_knn_k]

            if knn.shape[1] == 0:
                raise ValueError(
                    "train_geo_knn_npy has zero candidate columns after slicing")

            if n_anchors is not None:
                bad_mask = (knn < 0) | (knn >= int(n_anchors))
                if bad_mask.any():
                    bi, bj = np.argwhere(bad_mask)[0]
                    bad_val = int(knn[bi, bj])
                    raise ValueError(
                        "train_geo_knn_npy contains out-of-range anchor ids: "
                        f"value={bad_val} at row={bi}, col={bj}, allowed=[0,{int(n_anchors)-1}]"
                    )

            self.geo_knn = knn

        self.use_geo_knn = self.geo_knn is not None

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

        if self.use_geo_knn:
            knn_ids = self.geo_knn[idx]
        else:
            knn_ids = np.empty((0,), dtype=np.int64)

        return x, lat, lon, knn_ids


class AnchorModel(nn.Module):
    def __init__(self, backbone: str, proj_dim: int, use_xy_head: bool, learn_t: bool, init_t: float):
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

        self.learn_t = bool(learn_t)
        if self.learn_t:
            self.temp_raw = nn.Parameter(torch.tensor(
                inv_softplus(float(init_t)), dtype=torch.float32))
        else:
            self.register_parameter("temp_raw", None)

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        z = self.adapter(z)
        z = F.normalize(z, p=2, dim=1)
        xy = self.xy_head(z) if self.xy_head is not None else None
        return z, xy

    def get_temperature(self, fallback_t: float) -> torch.Tensor:
        if self.learn_t:
            return F.softplus(self.temp_raw) + 1e-6
        return torch.tensor(float(fallback_t), device=self.adapter[0].weight.device)


def make_optimizer(model: AnchorModel, lr_enc: float, lr_head: float, wd: float):
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]

    head_params = [p for p in model.adapter.parameters() if p.requires_grad]
    if model.xy_head is not None:
        head_params += [p for p in model.xy_head.parameters()
                        if p.requires_grad]
    if model.learn_t and model.temp_raw is not None and model.temp_raw.requires_grad:
        head_params += [model.temp_raw]

    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_enc, "name": "enc"})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "name": "head"})

    if not groups:
        raise RuntimeError("No trainable parameters found for optimizer")

    return torch.optim.AdamW(groups, weight_decay=wd)


def set_encoder_lr_cosine(opt: torch.optim.Optimizer, base_lr: float, min_lr: float, progress: float):
    # progress in [0,1]
    progress = float(max(0.0, min(1.0, progress)))
    cur = min_lr + 0.5 * (base_lr - min_lr) * \
        (1.0 + math.cos(math.pi * progress))
    for g in opt.param_groups:
        if g.get("name") == "enc":
            g["lr"] = cur


def build_pos_neg_and_target(
    d_m: torch.Tensor,
    pos_radius_m: float,
    neg_radius_m: float,
    tau_m: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # d_m: (B, M)
    bsz, m = d_m.shape

    pos_mask = d_m <= pos_radius_m
    neg_mask = d_m >= neg_radius_m

    closest_idx = torch.argsort(d_m, dim=1)
    k_pos_fallback = min(4, m)
    for i in range(bsz):
        if not pos_mask[i].any():
            pos_mask[i, closest_idx[i, :k_pos_fallback]] = True

    farthest_idx = torch.argsort(d_m, dim=1, descending=True)
    k_neg_fallback = min(16, m)
    for i in range(bsz):
        if not neg_mask[i].any():
            neg_mask[i, farthest_idx[i, :k_neg_fallback]] = True

    target = torch.zeros_like(d_m)
    scaled = -d_m / max(tau_m, 1e-6)

    for i in range(bsz):
        pm = pos_mask[i]
        logits_pos = scaled[i, pm]
        w_pos = F.softmax(logits_pos, dim=0).detach()
        target[i, pm] = w_pos

    pos_count = pos_mask.float().sum(dim=1)
    neg_count = neg_mask.float().sum(dim=1)
    return pos_mask, neg_mask, target, (pos_count, neg_count)


def main():
    ap = argparse.ArgumentParser(
        description="Anchor-bank StageB training with geo-topK candidates")
    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sep", default="auto")

    ap.add_argument("--anchors_vectors", required=True)
    ap.add_argument("--anchors_meta", required=True)

    ap.add_argument("--ckpt_init", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--freeze_epochs", type=int, default=5)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--topM", type=int, default=None)

    ap.add_argument("--topM_total", type=int, default=256)
    ap.add_argument("--pos_radius_m", type=float, default=900.0)
    ap.add_argument("--neg_radius_m", type=float, default=2000.0)
    ap.add_argument("--tau_m", type=float, default=450.0)

    ap.add_argument("--T", type=float, default=0.07)
    ap.add_argument("--Tpred", type=float, default=0.10)
    ap.add_argument("--learn_T", action="store_true")

    ap.add_argument("--lambda_geo", type=float, default=1.0)
    ap.add_argument("--lambda_ce", type=float, default=0.2)
    ap.add_argument("--huber_delta_m", type=float, default=300.0)

    # Optional XY head loss (off by default, normalized).
    ap.add_argument("--lambda_xy", type=float, default=0.0)
    ap.add_argument("--xy_scale", type=float, default=5000.0)
    ap.add_argument("--use_xy_head", action="store_true")

    # Geo-topK training candidates (precomputed by precompute_geo_knn.py)
    ap.add_argument("--train_geo_knn_npy", default="")
    ap.add_argument("--geo_knn_K", type=int, default=0)

    args = ap.parse_args()

    if args.batch_size > 16:
        raise ValueError("Use batch_size <= 16")

    top_m_total = int(args.topM if args.topM is not None else args.topM_total)
    if top_m_total <= 0:
        raise ValueError("topM_total must be > 0")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = vars(args).copy()
    cfg["topM_total_effective"] = top_m_total
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    anchors = np.load(args.anchors_vectors).astype(np.float32)
    if anchors.ndim != 2:
        raise ValueError(
            f"anchors_vectors must be shape [N,D], got {anchors.shape}")
    if anchors.shape[1] != int(args.proj_dim):
        raise ValueError(
            f"anchors_vectors dim mismatch: got D={anchors.shape[1]}, expected {args.proj_dim}")

    a_lat_np, a_lon_np, _ = load_anchor_meta(args.anchors_meta)
    if anchors.shape[0] != a_lat_np.shape[0]:
        raise ValueError(
            "anchors_vectors and anchors_meta mismatch: "
            f"vectors_rows={anchors.shape[0]}, meta_rows={a_lat_np.shape[0]}"
        )

    anchors = anchors / \
        np.clip(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-12, None)

    lat0 = float(np.mean(a_lat_np))
    lon0 = float(np.mean(a_lon_np))

    a_x_np, a_y_np = latlon_to_xy_m(a_lat_np, a_lon_np, lat0, lon0)
    a_xy_np = np.stack([a_x_np, a_y_np], axis=1).astype(np.float32)

    anchors_t = torch.from_numpy(anchors).to(device)
    a_lat_t = torch.from_numpy(a_lat_np.astype(np.float32)).to(device)
    a_lon_t = torch.from_numpy(a_lon_np.astype(np.float32)).to(device)
    a_xy_t = torch.from_numpy(a_xy_np).to(device)

    ds = AnchorTrainDataset(
        tiles_csv=args.tiles_csv,
        split=args.split,
        tiles_root=args.tiles_root,
        image_size=args.image_size,
        sep=args.sep,
        geo_knn_npy_path=args.train_geo_knn_npy,
        geo_knn_k=int(args.geo_knn_K),
        n_anchors=int(anchors.shape[0]),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

    model = AnchorModel(
        backbone=args.backbone,
        proj_dim=args.proj_dim,
        use_xy_head=bool(args.use_xy_head),
        learn_t=bool(args.learn_T),
        init_t=float(args.T),
    ).to(device)
    load_stageA_into_encoder(model.enc, args.ckpt_init)

    # Freeze encoder first epochs.
    for p in model.enc.parameters():
        p.requires_grad = False

    opt = make_optimizer(model, lr_enc=args.lr_enc,
                         lr_head=args.lr_head, wd=args.wd)
    huber_geo = nn.HuberLoss(delta=0.05)   # delta in normalized units
    huber_xy = nn.HuberLoss(delta=0.1)

    top_m_total = min(top_m_total, anchors.shape[0])

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for p in model.enc.parameters():
                p.requires_grad = True
            opt = make_optimizer(model, lr_enc=args.lr_enc,
                                 lr_head=args.lr_head, wd=args.wd)

        # Cosine LR for encoder only after unfreeze.
        if epoch > args.freeze_epochs and args.epochs > args.freeze_epochs:
            prog = (epoch - args.freeze_epochs - 1) / \
                max(1, (args.epochs - args.freeze_epochs - 1))
            set_encoder_lr_cosine(opt, base_lr=float(
                args.lr_enc), min_lr=float(args.lr_enc) * 0.1, progress=prog)

        model.train()
        pbar = tqdm(dl, desc=f"StageB(anchor) epoch {epoch}/{args.epochs}")

        run_total = 0.0
        run_geo = 0.0
        run_ce = 0.0
        run_med = 0.0
        run_mean = 0.0
        run_p90 = 0.0
        run_w250 = 0.0
        run_w500 = 0.0
        run_w1000 = 0.0
        run_w2000 = 0.0
        run_pos = 0.0
        run_neg = 0.0
        run_k = 0.0
        run_frac_pos = 0.0
        run_frac_neg = 0.0
        n = 0

        for x, lat_np, lon_np, knn_ids in pbar:
            x = x.to(device, non_blocking=True)
            lat = lat_np.to(device=device, dtype=torch.float32,
                            non_blocking=True)
            lon = lon_np.to(device=device, dtype=torch.float32,
                            non_blocking=True)

            qx_np, qy_np = latlon_to_xy_m(
                lat_np.numpy().astype(np.float64),
                lon_np.numpy().astype(np.float64),
                lat0,
                lon0,
            )
            q_xy = torch.from_numpy(
                np.stack([qx_np, qy_np], axis=1).astype(np.float32)).to(device)

            z, xy_head_pred = model(x)

            if ds.use_geo_knn:
                idx_top = knn_ids.to(
                    device=device, dtype=torch.long, non_blocking=True)
                if idx_top.ndim != 2 or idx_top.shape[1] == 0:
                    raise RuntimeError(
                        "Invalid geo-kNN batch: expected [B,K] with K>0. "
                        "Check --train_geo_knn_npy and --geo_knn_K."
                    )
                cand_anchors = anchors_t[idx_top]  # (B, K, D)
                sim_top = (z.unsqueeze(1) * cand_anchors).sum(dim=2)  # (B, K)
            else:
                sim = z @ anchors_t.T
                sim_top, idx_top = torch.topk(
                    sim, k=top_m_total, dim=1, largest=True)

            cand_lat = a_lat_t[idx_top]
            cand_lon = a_lon_t[idx_top]
            cand_xy = a_xy_t[idx_top]

            d_m = haversine_m_torch(lat.unsqueeze(
                1), lon.unsqueeze(1), cand_lat, cand_lon)

            pos_mask, neg_mask, target_w, (pos_count, neg_count) = build_pos_neg_and_target(
                d_m=d_m,
                pos_radius_m=float(args.pos_radius_m),
                neg_radius_m=float(args.neg_radius_m),
                tau_m=float(args.tau_m),
            )

            t_cur = model.get_temperature(float(args.T))
            loss_ce = -(target_w * F.log_softmax(sim_top /
                        t_cur, dim=1)).sum(dim=1).mean()

            # Main geo prediction from weighted anchor coordinates.
            w_pred = F.softmax(sim_top / float(args.Tpred), dim=1)
            pred_xy = (w_pred.unsqueeze(2) * cand_xy).sum(dim=1)
            scale = float(args.xy_scale)

            if scale <= 0.0:
                scale = float(np.std(a_xy_np, axis=0).mean())
                scale = max(scale, 1.0)

            pred_xy_n = pred_xy / scale
            q_xy_n = q_xy / scale
            loss_geo = huber_geo(pred_xy_n, q_xy_n)

            # Optional small normalized XY-head loss.
            loss_xy = torch.tensor(0.0, device=device)
            if args.lambda_xy > 0.0 and xy_head_pred is not None:
                scale = float(args.xy_scale)
                if scale <= 0.0:
                    scale = float(np.std(a_xy_np, axis=0).mean())
                    scale = max(scale, 1.0)
                loss_xy = huber_xy(xy_head_pred / scale, q_xy / scale)

            loss = float(args.lambda_geo) * loss_geo + float(args.lambda_ce) * \
                loss_ce + float(args.lambda_xy) * loss_xy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred_lat, pred_lon = xy_to_latlon_m(
                    pred_xy[:, 0], pred_xy[:, 1], lat0, lon0)
                dist_m = haversine_m_torch(lat, lon, pred_lat, pred_lon)

                run_total += float(loss.item())
                run_geo += float(loss_geo.item())
                run_ce += float(loss_ce.item())
                run_med += float(torch.median(dist_m).item())
                run_mean += float(torch.mean(dist_m).item())
                run_p90 += float(torch.quantile(dist_m, 0.90).item())
                run_w250 += float((dist_m <= 250.0).float().mean().item())
                run_w500 += float((dist_m <= 500.0).float().mean().item())
                run_w1000 += float((dist_m <= 1000.0).float().mean().item())
                run_w2000 += float((dist_m <= 2000.0).float().mean().item())
                run_pos += float(pos_count.mean().item())
                run_neg += float(neg_count.mean().item())
                run_k += float(idx_top.shape[1])
                run_frac_pos += float(pos_mask.float().mean().item())
                run_frac_neg += float(neg_mask.float().mean().item())
                n += 1

            pbar.set_postfix(
                total=run_total / n,
                loss_geo=run_geo / n,
                loss_ce=run_ce / n,
                median_m=run_med / n,
                mean_m=run_mean / n,
                p90_m=run_p90 / n,
                w250=run_w250 / n,
                w500=run_w500 / n,
                w1000=run_w1000 / n,
                w2000=run_w2000 / n,
                pos=run_pos / n,
                neg=run_neg / n,
                avg_K=run_k / n,
                frac_pos=run_frac_pos / n,
                frac_neg=run_frac_neg / n,
                frozen=(epoch <= args.freeze_epochs),
            )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "lat0": lat0,
                "lon0": lon0,
            },
            os.path.join(args.out_dir, f"stageB_anchor_epoch{epoch:03d}.pt"),
        )

    torch.save(
        {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "lat0": lat0,
            "lon0": lon0,
        },
        os.path.join(args.out_dir, "stageB_anchor_last.pt"),
    )


if __name__ == "__main__":
    main()
