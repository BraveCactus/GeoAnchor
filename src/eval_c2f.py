import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_stageB_c2f import TileC2FDataset, GeoC2FModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_csv", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sep", default="auto")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid_n", type=int, default=12)
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = TileC2FDataset(
        tiles_csv=args.tiles_csv,
        split=args.split,
        tiles_root=args.tiles_root,
        image_size=args.image_size,
        grid_n=args.grid_n,
        sep=args.sep,
        seed=42,
        easy_aug=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = args.grid_n * args.grid_n
    model = GeoC2FModel(args.backbone, args.proj_dim, num_classes).to(dev).eval()

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd, strict=True)

    # утилиты: cell center -> xy, residual -> meters
    xmin,xmax,ymin,ymax = ds.xmin, ds.xmax, ds.ymin, ds.ymax
    cell_w, cell_h = ds.cell_w, ds.cell_h
    rx_scale, ry_scale = ds.rx_scale, ds.ry_scale

    def cls_to_center(cls):
        cx = cls % args.grid_n
        cy = cls // args.grid_n
        center_x = xmin + (cx + 0.5) * cell_w
        center_y = ymin + (cy + 0.5) * cell_h
        return center_x, center_y

    dists = []
    with torch.no_grad():
        for v1, v2, cls_true, res_true in tqdm(dl, desc="eval-c2f"):
            v1 = v1.to(dev, non_blocking=True)
            _, logits, res = model(v1)
            pred_cls = logits.argmax(dim=1).cpu().numpy()

            res = res.cpu().numpy()
            px, py = cls_to_center(pred_cls)
            px = px + res[:,0] * rx_scale
            py = py + res[:,1] * ry_scale

            cls_true = cls_true.numpy()
            res_true = res_true.numpy()
            tx, ty = cls_to_center(cls_true)
            tx = tx + res_true[:,0] * rx_scale
            ty = ty + res_true[:,1] * ry_scale

            dist = np.sqrt((px-tx)**2 + (py-ty)**2)
            dists.append(dist)

    d = np.concatenate(dists)
    out = {
        "n": int(d.size),
        "median_m": float(np.median(d)),
        "mean_m": float(np.mean(d)),
        "p90_m": float(np.quantile(d, 0.90)),
        "p95_m": float(np.quantile(d, 0.95)),
        "within_250m": float((d<=250).mean()),
        "within_500m": float((d<=500).mean()),
        "within_1000m": float((d<=1000).mean()),
        "within_2000m": float((d<=2000).mean()),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
