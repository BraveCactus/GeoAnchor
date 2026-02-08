import torch
import torch.nn.functional as F


def ntxent_loss(z1, z2, temperature=0.1):

    # Если что, то тут мы ожидаем, что z1,z2 уже нормализованы (B, D)

    device = z1.device
    B = z1.size(0)

    # склеиваем
    z = torch.cat([z1, z2], dim=0)      # (2B, D)
    sim = (z @ z.t()) / temperature     # (2B, 2B)

    # убираем совпадение с самим собой
    mask = torch.eye(2 * B, device=device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # правильные пары: i и i+B
    pos = torch.cat([
        torch.arange(B, 2 * B, device=device),
        torch.arange(0, B, device=device)
    ], dim=0)

    loss = F.cross_entropy(sim, pos)
    return loss
