from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, image):
        if not isinstance(self.base_transform, list):
            # return [self.base_transform(image) for _ in range(self.n_views)]
            return {
                "image": [
                    self.base_transform(image=image)["image"]
                    for _ in range(self.n_views)
                ]
            }
        else:
            # return [self.base_transform[i](image) for i in range(self.n_views)]
            return {
                "image": [
                    self.base_transform[i](image=image)["image"]
                    for i in range(self.n_views)
                ]
            }


# ================= SupConLoss =================
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, device="cuda"):
        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class TALONModel(nn.Module):
    def __init__(
        self,
        v_dim: int,
        known_num_classes: int,
        backbone,
        type: Literal["clip", "dino"] = "clip",
    ):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone_type = type
        if type == "clip":
            if (
                hasattr(self.backbone.visual, "transformer")
                and len(self.backbone.visual.transformer.resblocks) > 0
            ):
                for p in self.backbone.visual.transformer.resblocks[-1].parameters():
                    p.requires_grad = True
            if hasattr(self.backbone.visual, "ln_post"):
                for p in self.backbone.visual.ln_post.parameters():
                    p.requires_grad = True
            if (
                hasattr(self.backbone.visual, "proj")
                and self.backbone.visual.proj is not None
            ):
                try:
                    self.backbone.visual.proj.requires_grad_(True)
                except Exception:
                    pass
        else:
            self.backbone.blocks[-1].requires_grad_(True)

        self.proto = nn.Parameter(torch.randn(known_num_classes, v_dim))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "clip":
            v = self.backbone.encode_image(images).float()
        else:
            v = self.backbone(images)
        return F.normalize(v, dim=-1)

    def forward(self, images: torch.Tensor, targets=None) -> torch.Tensor:
        v_emb = self.encode_image(images)  # [B, Dv])
        return v_emb

    @torch.no_grad()
    def save_visual_proj(self):
        return {k: v.clone() for k, v in self.backbone.state_dict().items()}

    @torch.no_grad()
    def set_visual_proj(self, state_dict: dict):
        self.backbone.load_state_dict(state_dict)
