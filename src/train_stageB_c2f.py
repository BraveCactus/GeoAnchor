import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import tifffile as tiff
from torchvision import transforms

from data.augs import build_transforms
from model.dinov2_gem import DinoV2Encoder


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


def latlon_to_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    lat0r = np.deg2rad(lat0)
    x = (lon - lon0) * np.cos(lat0r) * 111320.0
    y = (lat - lat0) * 110540.0
    return x, y


class TileC2FDataset(Dataset):
    """
    patch_name only. returns:
      v1, v2, cls_id, residual_norm(2)
    """

    def __init__(
        self,
        tiles_csv: str,
        split: str,
        tiles_root: str,
        image_size: int,
        grid_n: int,
        sep: str = "auto",
        exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        seed: int = 42,
        # IMPORTANT: if True, use same "tile" transforms for both views (easier at start)
        easy_aug: bool = True,
    ):
        import pandas as pd
        if sep == "auto":
            df = pd.read_csv(tiles_csv, sep=None, engine="python")
        else:
            sep_real = "\t" if sep == "\\t" else sep
            df = pd.read_csv(tiles_csv, sep=sep_real)

        if "split" not in df.columns:
            raise ValueError("CSV должен содержать split")
        if "patch_name" not in df.columns:
            raise ValueError("CSV должен содержать patch_name")
        if not {"lat", "lon"}.issubset(df.columns):
            raise ValueError("CSV должен содержать lat, lon")

        df = df[df["split"].astype(str) == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Нет строк для split='{split}'")

        self.df = df
        self.tiles_root = os.path.abspath(tiles_root)
        self.exts = exts
        self.grid_n = int(grid_n)

        # Augs:
        # На старте классификации важно не "ломать" тайл.
        # Поэтому первые эпохи лучше делать easy_aug=True: обе вьюхи mode="tile".
        self.t1 = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.t2 = self.t1

        lat = df["lat"].astype(float).to_numpy()
        lon = df["lon"].astype(float).to_numpy()
        self.lat0 = float(lat.mean())
        self.lon0 = float(lon.mean())

        x_m, y_m = latlon_to_xy_m(lat, lon, self.lat0, self.lon0)
        self.x_m = x_m.astype(np.float32)
        self.y_m = y_m.astype(np.float32)

        # bounds (train split)
        self.xmin = float(self.x_m.min()) - 1e-3
        self.xmax = float(self.x_m.max()) + 1e-3
        self.ymin = float(self.y_m.min()) - 1e-3
        self.ymax = float(self.y_m.max()) + 1e-3

        self.cell_w = (self.xmax - self.xmin) / self.grid_n
        self.cell_h = (self.ymax - self.ymin) / self.grid_n

        self.rx_scale = float(self.cell_w / 2.0 + 1e-6)
        self.ry_scale = float(self.cell_h / 2.0 + 1e-6)

        self.rng = np.random.default_rng(seed)

    def _infer_path(self, patch_name: str) -> str:
        base = os.path.join(self.tiles_root, patch_name)
        for ext in self.exts:
            p = base + ext
            if os.path.exists(p):
                return p
        return base + self.exts[0]

    def _xy_to_cell(self, x: float, y: float):
        cx = int((x - self.xmin) / self.cell_w)
        cy = int((y - self.ymin) / self.cell_h)
        cx = max(0, min(self.grid_n - 1, cx))
        cy = max(0, min(self.grid_n - 1, cy))
        cls = cy * self.grid_n + cx

        center_x = self.xmin + (cx + 0.5) * self.cell_w
        center_y = self.ymin + (cy + 0.5) * self.cell_h
        rx = x - center_x
        ry = y - center_y
        return cls, rx, ry

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        patch_name = str(self.df.iloc[idx]["patch_name"])
        p = self._infer_path(patch_name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Не найден файл: {p}")

        img = open_image_any(p)
        v1 = self.t1(img)
        v2 = self.t2(img)

        x = float(self.x_m[idx])
        y = float(self.y_m[idx])
        cls, rx, ry = self._xy_to_cell(x, y)

        res = np.array([rx / self.rx_scale, ry /
                       self.ry_scale], dtype=np.float32)
        return v1, v2, np.int64(cls), res


class GeoC2FModel(nn.Module):
    def __init__(self, backbone: str, proj_dim: int, num_classes: int):
        super().__init__()
        self.enc = DinoV2Encoder(backbone_name=backbone, proj_dim=proj_dim)
        self.cls_head = nn.Linear(proj_dim, num_classes)
        self.res_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, 2),
        )

    def forward(self, x):
        z = self.enc(x)
        return z, self.cls_head(z), self.res_head(z)


def make_optimizer(model: GeoC2FModel, lr_enc: float, lr_head: float, wd: float):
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    head_params = list(model.cls_head.parameters()) + \
        list(model.res_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr_enc},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=wd,
    )


def make_neighbor_ids_and_weights(cls: torch.Tensor, grid_n: int, main_w: float = 0.6):
    """
    ids: (B,5) = [self, up, down, left, right], invalid=-1
    w:   (B,5) normalized over valid ids (borders/corners handled)
    """
    device = cls.device
    B = cls.shape[0]

    cx = cls % grid_n
    cy = cls // grid_n

    ids = torch.full((B, 5), -1, dtype=torch.long, device=device)
    ids[:, 0] = cls

    up = cy > 0
    ids[up, 1] = (cy[up] - 1) * grid_n + cx[up]

    down = cy < (grid_n - 1)
    ids[down, 2] = (cy[down] + 1) * grid_n + cx[down]

    left = cx > 0
    ids[left, 3] = cy[left] * grid_n + (cx[left] - 1)

    right = cx < (grid_n - 1)
    ids[right, 4] = cy[right] * grid_n + (cx[right] + 1)

    # weights: main on self, rest equally split among VALID neighbors only
    w = torch.zeros((B, 5), dtype=torch.float32, device=device)
    w[:, 0] = main_w

    neigh_valid = (ids[:, 1:] >= 0)
    neigh_cnt = neigh_valid.sum(dim=1).clamp(min=1).float()
    w_neigh = (1.0 - main_w) / neigh_cnt
    w[:, 1:] = neigh_valid.float() * w_neigh.unsqueeze(1)

    # normalize over valid ids (safety)
    valid = ids >= 0
    w = w * valid.float()
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)

    return ids, w


def soft_ce_neighbors(logits: torch.Tensor, cls: torch.Tensor, grid_n: int, main_w: float = 0.6) -> torch.Tensor:
    ids, w = make_neighbor_ids_and_weights(cls, grid_n, main_w=main_w)
    logp = F.log_softmax(logits, dim=1)

    ids_safe = ids.clamp(min=0)
    gathered = logp.gather(1, ids_safe)
    gathered = gathered * (ids >= 0).float()

    return -(w * gathered).sum(dim=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--sep", default="auto")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_init", required=True)

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")

    # coarse-to-fine
    ap.add_argument("--grid_n", type=int, default=12)
    ap.add_argument("--lambda_cls", type=float, default=2.0)
    ap.add_argument("--lambda_res", type=float, default=1.0)
    ap.add_argument("--huber_delta", type=float, default=0.2)
    ap.add_argument("--neighbor_main_w", type=float, default=0.6,
                    help="Main-cell probability mass for neighbor soft CE (remaining mass goes to 4-neighbors).")

    # schedule
    ap.add_argument("--freeze_epochs", type=int, default=10)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--lr_enc", type=float, default=2e-5)

    args = ap.parse_args()
    set_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config_c2f.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # easy aug for initial stage
    ds = TileC2FDataset(
        tiles_csv=args.tiles_csv,
        split=args.split,
        tiles_root=args.tiles_root,
        image_size=args.image_size,
        grid_n=args.grid_n,
        sep=args.sep,
        seed=args.seed,
        easy_aug=True,
    )
    num_classes = args.grid_n * args.grid_n

    with open(os.path.join(args.out_dir, "xy_grid.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "lat0": ds.lat0, "lon0": ds.lon0,
                "xmin": ds.xmin, "xmax": ds.xmax,
                "ymin": ds.ymin, "ymax": ds.ymax,
                "grid_n": ds.grid_n,
                "cell_w": ds.cell_w, "cell_h": ds.cell_h,
                "rx_scale": ds.rx_scale, "ry_scale": ds.ry_scale,
            },
            f, ensure_ascii=False, indent=2
        )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

    model = GeoC2FModel(args.backbone, args.proj_dim, num_classes).to(dev)

    # init encoder from stageA
    ckpt = torch.load(args.ckpt_init, map_location="cpu")
    model.enc.load_state_dict(ckpt["model"], strict=True)
    model.train()

    # freeze encoder first
    for p in model.enc.parameters():
        p.requires_grad = False

    opt = make_optimizer(model, lr_enc=args.lr_enc,
                         lr_head=args.lr_head, wd=args.wd)
    huber = nn.HuberLoss(delta=args.huber_delta)

    for epoch in range(1, args.epochs + 1):
        # unfreeze encoder after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            for p in model.enc.parameters():
                p.requires_grad = True
            opt = make_optimizer(model, lr_enc=args.lr_enc,
                                 lr_head=args.lr_head, wd=args.wd)

        pbar = tqdm(dl, desc=f"StageB(c2f-freeze) epoch {epoch}/{args.epochs}")

        run_loss = 0.0
        run_cls = 0.0
        run_res = 0.0
        run_acc = 0.0
        run_nacc = 0.0
        n = 0

        for v1, v2, cls, res in pbar:
            v1 = v1.to(dev, non_blocking=True)
            v2 = v2.to(dev, non_blocking=True)
            cls = cls.to(dev, non_blocking=True)
            res = res.to(dev, non_blocking=True)

            _, logits1, res1 = model(v1)
            _, logits2, res2 = model(v2)

            main_w = args.neighbor_main_w
            loss_cls = 0.5 * soft_ce_neighbors(logits1, cls, args.grid_n, main_w=main_w) + \
                0.5 * soft_ce_neighbors(logits2, cls,
                                        args.grid_n, main_w=main_w)
            loss_res = 0.5 * huber(res1, res) + 0.5 * huber(res2, res)
            loss = args.lambda_cls * loss_cls + args.lambda_res * loss_res

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits1.argmax(dim=1)
                acc = (pred == cls).float().mean().item()
                ids, _ = make_neighbor_ids_and_weights(
                    cls, args.grid_n, main_w=main_w)
                nacc = (pred.unsqueeze(1) == ids).any(
                    dim=1).float().mean().item()

            run_loss += float(loss.item())
            run_cls += float(loss_cls.item())
            run_res += float(loss_res.item())
            run_acc += float(acc)
            run_nacc += float(nacc)
            n += 1

            pbar.set_postfix(
                loss=run_loss / n,
                cls=run_cls / n,
                res=run_res / n,
                acc=run_acc / n,
                nacc=run_nacc / n,
                frozen=(epoch <= args.freeze_epochs),
            )

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
             "args": vars(args)},
            os.path.join(args.out_dir, f"stageB_c2f_epoch{epoch:03d}.pt"),
        )

    torch.save(
        {"epoch": args.epochs, "model": model.state_dict(
        ), "opt": opt.state_dict(), "args": vars(args)},
        os.path.join(args.out_dir, "stageB_c2f_last.pt"),
    )


if __name__ == "__main__":
    main()
