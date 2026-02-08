import os
import argparse
import numpy as np
import torch

from model.dinov2_gem import DinoV2Encoder


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--proj_dim", type=int, default=512)

    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # грузим модель
    model = DinoV2Encoder(backbone_name=args.backbone,
                          proj_dim=args.proj_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # dummy прогон, чтоб не совсем вслепую экспортить
    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    torch_out = model(dummy).cpu().numpy()

    onnx_path = os.path.join(args.out_dir, "encoder.onnx")

    # экспорт в onnx
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        dynamic_axes={
            "image": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )

    # мини sanity check: просто повторяем torch прогон (onnx runtime тут не проверяем)
    torch_out2 = model(dummy).cpu().numpy()
    diff = float(np.max(np.abs(torch_out - torch_out2)))

    # косинус, чисто убедиться что одинаково
    cos = float(
        np.sum(torch_out * torch_out2) /
        (np.linalg.norm(torch_out) * np.linalg.norm(torch_out2) + 1e-9)
    )

    print("Saved:", onnx_path)
    print("Torch self-check max|diff|:", diff, "cos:", cos)


if __name__ == "__main__":
    main()
