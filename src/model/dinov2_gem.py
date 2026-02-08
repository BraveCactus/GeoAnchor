import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    # типа generalized mean pooling по токенам (B, T, C) -> (B, C)
    def __init__(self, p=3.0, eps=1e-6, trainable=True):
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, T, C)
        p = torch.clamp(self.p, min=0.1, max=10.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=1).pow(1.0 / p)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DinoV2Encoder(nn.Module):
    # DINOv2 (ViT) + pooling (GeM) + proj head + L2 normalize
    # написано “по-быстрому”, но должно нормально работать

    def __init__(
        self,
        backbone_name="dinov2_vits14",
        proj_dim=512,
        gem_p=3.0,
        use_cls=False,
        train_gem=True,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # тянем модель из torch.hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name)
        self.backbone.eval()  # по дефолту в eval, в train-скрипте можно включить train()

        self.use_cls = use_cls
        self.gem = GeM(p=gem_p, trainable=train_gem)

        # надо понять размерность выходных токенов (типа 384/768 и тд)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            tokens = self._forward_tokens(dummy)
            in_dim = tokens.shape[-1]

        self.proj = ProjectionHead(
            in_dim=in_dim,
            out_dim=proj_dim,
            hidden_dim=max(1024, proj_dim * 2),
        )
        self.out_dim = proj_dim

    def _forward_tokens(self, x):
        # возвращаем последовательность токенов для пула: (B, T, C)
        out = self.backbone.forward_features(x)
        # - 'x_norm_clstoken' (B, C)
        # - 'x_norm_patchtokens' (B, T, C)
        if "x_norm_patchtokens" in out:
            patch = out["x_norm_patchtokens"]
            if self.use_cls and "x_norm_clstoken" in out:
                cls = out["x_norm_clstoken"].unsqueeze(1)  # (B, 1, C)
                return torch.cat([cls, patch], dim=1)
            return patch

        # запасной вариант: вдруг вернули токены в 'x'
        if "x" in out and getattr(out["x"], "dim", lambda: 0)() == 3:
            return out["x"]

        raise RuntimeError(
            f"непонятный формат выхода forward_features, keys={list(out.keys())}")

    def forward(self, x):
        tokens = self._forward_tokens(x)   # (B, T, C)
        pooled = self.gem(tokens)         # (B, C)
        z = self.proj(pooled)             # (B, D)
        z = F.normalize(z, p=2, dim=-1)   # просто L2 нормализация
        return z
