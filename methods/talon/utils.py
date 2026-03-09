import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tools.evaluate_utils import cluster_acc


@torch.no_grad()
def build_ncm_prototypes(loader, model, device, known_classes):
    model.eval()
    K = len(known_classes)
    D = int(model.proto.size(1))
    sums = torch.zeros(K, D, device=device)
    counts = torch.zeros(K, dtype=torch.long, device=device)

    for images, label, *rest in tqdm(loader, desc="NCM"):
        images = images.to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            f = model(images)

        m = f.float()

        # 累加特征
        sums.index_add_(0, label, m)
        # 累加计数
        counts.index_add_(0, label, torch.ones_like(label))

    P = sums
    nz = counts > 0
    if nz.any():
        P[nz] = F.normalize(P[nz] / counts[nz].unsqueeze(1), dim=-1)
    if (~nz).any():
        fallback = F.normalize(
            model.proto.data.to(device)[(~nz).nonzero(as_tuple=True)[0]], dim=-1
        )
        P[(~nz)] = fallback
    return P  # [K, D]


def angle_logits(
    x: torch.Tensor,
    W: torch.Tensor,
    s: float = 32,
    m: float = 0.3,
    eps: float = 1e-6,
):
    x = F.normalize(x.float(), dim=-1)
    W = F.normalize(W.float(), dim=-1)

    cosine = torch.matmul(x, W.t()).clamp(-1.0, 1.0)  # [B, K]
    sin_theta = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(eps))
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    phi = cosine * cos_m - sin_theta * sin_m  # cos(theta + m)

    th = math.cos(math.pi - m)
    mm = math.sin(math.pi - m) * m
    phi = torch.where(cosine > th, phi, cosine - mm)

    return s * phi, s * cosine


def split_cluster_acc_v1(y_true, y_pred, mask):
    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    if y_pred.size == 0:
        return 0.0, 0.0, 0.0

    weight = mask.mean()

    old_size = int(mask.sum())
    new_size = int((~mask).sum())
    old_acc = cluster_acc(y_true[mask], y_pred[mask]) if old_size > 0 else 0.0
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask]) if new_size > 0 else 0.0
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc
