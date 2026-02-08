import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from PIL import Image
import tifffile as tiff

from data.augs import build_transforms
from model.dinov2_gem import DinoV2Encoder
from losses.ntxent import ntxent_loss


# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: str, model, optimizer, epoch: int, args, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def resolve_path(p: str, base_dir: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.isabs(p) and not os.path.exists(p):
        return os.path.join(base_dir, os.path.basename(p))
    return os.path.join(base_dir, p)


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


# ---------------------------
# Dataset (new CSV)
# ---------------------------

class NewCsvSimCLRDataset(Dataset):
    """
    CSV:
      patch_name,image_name,tile_path,lat,lon,split,embedding...

    Для StageA embedding-колонки НЕ используются.

    Как ищется файл:
      1) если задан --image_col и колонка есть в CSV -> берём путь оттуда
      2) иначе ищем tiles_root/<patch_name>.(tif/tiff/png/jpg/...)
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        transform,
        tiles_root: str,
        image_col: str = "",
        exts: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    ):
        import pandas as pd

        self.df = pd.read_csv(csv_path)

        if "split" not in self.df.columns:
            raise ValueError("CSV должен содержать колонку split")
        if "patch_name" not in self.df.columns:
            raise ValueError("CSV должен содержать колонку patch_name")

        self.df = self.df[self.df["split"].astype(
            str) == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Нет строк для split='{split}' в {csv_path}")

        self.transform = transform
        self.tiles_root = tiles_root
        self.image_col = image_col.strip()
        self.exts = exts
        self.use_csv_path = bool(self.image_col) and (
            self.image_col in self.df.columns)

    def _infer_tile_image_path(self, patch_name: str) -> str:
        base = os.path.join(self.tiles_root, patch_name)
        for ext in self.exts:
            p = base + ext
            if os.path.exists(p):
                return p
        return base + self.exts[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patch_name = str(row["patch_name"])

        if self.use_csv_path:
            p_raw = str(row[self.image_col])
            img_path = resolve_path(p_raw, self.tiles_root)
        else:
            img_path = self._infer_tile_image_path(patch_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Не найден файл тайла: {img_path}\n"
                f"Ожидалось: tiles_root/<patch_name>.tif(.tiff) или указать --image_col с путём к файлу."
            )

        img = open_image_any(img_path)
        v1 = self.transform(img)
        v2 = self.transform(img)
        return v1, v2


# ---------------------------
# Eval (validation loss)
# ---------------------------

@torch.no_grad()
def eval_val_loss(model, dl, device, temperature: float) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    for v1, v2 in dl:
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        z1 = model(v1)
        z2 = model(v2)
        loss = ntxent_loss(z1, z2, temperature=temperature)
        total += float(loss.item())
        n_batches += 1
    model.train()
    return total / max(1, n_batches)


# ---------------------------
# Train Stage A
# ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--image_col", default="")

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)  # кратно 14!
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")

    # validation config
    ap.add_argument(
        "--val_mode",
        choices=["from_csv", "subset", "none"],
        default="subset",
        help="from_csv: use split=='val' if present; subset: fixed subset from train; none: no val",
    )
    ap.add_argument("--val_fraction", type=float, default=0.05,
                    help="For subset: fraction of train")
    ap.add_argument("--val_max_items", type=int,
                    default=2000, help="Cap subset size")

    args = ap.parse_args()
    set_seed(args.seed)

    # project root for relative paths
    this_file = os.path.abspath(__file__)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(this_file), ".."))

    tiles_root = args.tiles_root
    if not os.path.isabs(tiles_root):
        tiles_root = os.path.join(project_root, tiles_root)
    tiles_root = os.path.abspath(tiles_root)

    if not os.path.isdir(tiles_root):
        raise FileNotFoundError(
            f"--tiles_root не существует или не папка: {tiles_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tfm_train = build_transforms(args.image_size, mode="stageA")
    ds_train = NewCsvSimCLRDataset(
        csv_path=str(args.tiles_csv),
        split="train",
        transform=tfm_train,
        tiles_root=tiles_root,
        image_col=args.image_col,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # --- build val loader (ALWAYS used each epoch unless val_mode=none) ---
    dl_val = None
    if args.val_mode != "none":
        tfm_val = build_transforms(args.image_size, mode="eval")

        if args.val_mode == "from_csv":
            try:
                ds_val = NewCsvSimCLRDataset(
                    csv_path=str(args.tiles_csv),
                    split="val",
                    transform=tfm_val,
                    tiles_root=tiles_root,
                    image_col=args.image_col,
                )
            except Exception:
                # fallback to subset if val split doesn't exist
                args.val_mode = "subset"
                ds_val = None
        else:
            ds_val = None

        if args.val_mode == "subset":
            ds_train_for_val = NewCsvSimCLRDataset(
                csv_path=str(args.tiles_csv),
                split="train",
                transform=tfm_val,
                tiles_root=tiles_root,
                image_col=args.image_col,
            )
            n = len(ds_train_for_val)
            val_n = int(max(1, n * args.val_fraction))
            val_n = min(val_n, args.val_max_items)

            rng = np.random.default_rng(args.seed)
            idxs = rng.choice(n, size=val_n, replace=False).tolist()
            ds_val = Subset(ds_train_for_val, idxs)

        dl_val = DataLoader(
            ds_val,
            batch_size=min(args.batch_size, 64),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

    model = DinoV2Encoder(backbone_name=args.backbone,
                          proj_dim=args.proj_dim).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = vars(args).copy()
    cfg["project_root"] = project_root
    cfg["tiles_root_abs"] = tiles_root
    with open(os.path.join(args.out_dir, "config_stageA.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    metrics_path = os.path.join(args.out_dir, "metrics_stageA.jsonl")
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dl_train, desc=f"StageA epoch {epoch}/{args.epochs}")
        running = 0.0

        for v1, v2 in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            z1 = model(v1)
            z2 = model(v2)

            loss = ntxent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            pbar.set_postfix(train_loss=running / max(1, (pbar.n + 1)))

        train_loss = running / max(1, len(dl_train))

        # ---- VALIDATION AFTER EACH EPOCH ----
        val_loss = None
        if dl_val is not None:
            val_loss = eval_val_loss(
                model, dl_val, device, temperature=args.temperature)

        # save epoch checkpoint
        save_ckpt(
            os.path.join(args.out_dir, f"stageA_epoch{epoch:03d}.pt"),
            model,
            optimizer,
            epoch,
            args,
            extra={"train_loss": train_loss, "val_loss": val_loss},
        )

        # save best checkpoint by val loss
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_ckpt(
                os.path.join(args.out_dir, "stageA_best.pt"),
                model,
                optimizer,
                epoch,
                args,
                extra={"train_loss": train_loss, "val_loss": val_loss},
            )

        # log metrics line
        rec = {"epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            rec["val_loss"] = val_loss
            rec["best_val"] = best_val
            rec["best_epoch"] = best_epoch
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    save_ckpt(
        os.path.join(args.out_dir, "stageA_last.pt"),
        model,
        optimizer,
        args.epochs,
        args,
        extra={"best_val": best_val, "best_epoch": best_epoch},
    )


if __name__ == "__main__":
    main()
