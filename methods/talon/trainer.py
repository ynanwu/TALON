from typing import Literal, Optional

import clip
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from config import pretrain_path
from methods.talon.model import (
    ContrastiveLearningViewGenerator,
    SupConLoss,
    TALONModel,
)
from methods.talon.utils import (
    angle_logits,
    build_ncm_prototypes,
    split_cluster_acc_v1,
)
from tools.evaluate_utils import cluster_acc, split_cluster_acc
from tools.train_utils import SmoothedValue


class Trainer:
    def __init__(
        self,
        tau: float,
        tta_state: Literal["M", "P", "M+P", None],
        epochs: int,
        train_classes: list,
        unlabel_classes: list,
        device,
        start_epoch: int = 0,
        max_norm: Optional[float] = None,
        encoder: Literal["dino", "clip"] = "clip",
        checkpoint_dir: Optional[str] = None,
    ):
        self.tau = tau
        self.tta_state = tta_state
        self.epochs = epochs
        self.train_classes = train_classes
        self.unlabel_classes = unlabel_classes
        self.device = device
        self.start_epoch = start_epoch
        self.max_norm = max_norm
        self.supcon_t = 0.08
        self.arc_s = 30.0
        self.arc_m = 0.2
        self.max_norm = None
        self.checkpoint_dir = checkpoint_dir
        self.model = self._build_model(encoder=encoder)
        self._build_model_tta(encoder)

        backbone_params = []
        proto_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            elif "proto" in name:
                proto_params.append(param)
            else:
                raise ValueError(f"Unknown parameter group: {name}")
        param_group = [
            {"params": backbone_params, "weight_decay": 1e-3},
            {"params": proto_params, "weight_decay": 1e-3},
        ]
        self.optimizer = AdamW(param_group)

        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.epochs,
            warmup_lr_init=1e-4,
            warmup_t=50,
            cycle_limit=1,
            t_in_epochs=True,
        )

        self.scaler = GradScaler()
        self.tta_scaler = GradScaler()

    def _build_model(self, encoder: Literal["dino", "clip"]):
        if encoder == "clip":
            clip_model, _ = clip.load("ViT-B/16", device=self.device)
            clip_model = clip_model.float()
            model = TALONModel(
                v_dim=512,
                known_num_classes=len(self.train_classes),
                backbone=clip_model,
                type="clip",
            )
        else:
            dino_model = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=0
            )
            state_dict = torch.load(pretrain_path, map_location="cpu")
            dino_model.load_state_dict(state_dict, strict=True)
            model = TALONModel(
                v_dim=768,
                known_num_classes=len(self.train_classes),
                backbone=dino_model,
                type="dino",
            )
        return model.to(self.device)

    def _build_model_tta(self, enc: Literal["dino", "clip"]):
        if enc == "clip":
            block = self.model.backbone.visual.transformer.resblocks[-1]
        else:
            block = self.model.backbone.blocks[-1]
        self.model_tta_params = [
            p
            for m in block.modules()
            if isinstance(
                m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)
            )
            for name, p in m.named_parameters(recurse=False)
            if name in ["weight", "bias"] and p.requires_grad
        ]
        if enc == "clip":
            self.model_tta_optimizer = torch.optim.SGD(self.model_tta_params, lr=1e-1)
        else:
            self.model_tta_optimizer = torch.optim.SGD(self.model_tta_params, lr=1e-2)

    def model_tta(
        self,
        images,
        y_inds,
        P,
    ):
        was_training = self.model.training
        try:
            self.model.eval()

            Pn = F.normalize(P.detach(), dim=-1)

            with torch.enable_grad():
                with torch.autocast(device_type="cuda"):
                    feat = self.model(images)
                m = feat.float()
                m = F.normalize(m, dim=-1)
                logits = 30 * (m @ Pn.t()).float()

                m_u = m
                y_u = y_inds

                log_probs = F.log_softmax(logits, dim=1)
                loss_ent = -(log_probs.exp() * log_probs).sum(dim=1).mean(0)

                unique_labels, inverse_indices = torch.unique(
                    y_u, sorted=True, return_inverse=True
                )
                L = unique_labels.size(0)
                D = m_u.size(1)

                # 1. 计算每个类别的特征和 (Scatter Add)
                # 初始化 [L, D]
                sums = torch.zeros(L, D, device=m_u.device, dtype=m_u.dtype)
                # index 需要扩展维度以匹配 scatter
                idx_expanded = inverse_indices.view(-1, 1).expand(-1, D)
                sums.scatter_add_(0, idx_expanded, m_u)

                # 2. 计算每个类别的计数
                counts = (
                    torch.bincount(inverse_indices, minlength=L).float().unsqueeze(1)
                )

                # 3. 计算均值并归一化
                # [L, D]
                class_means = F.normalize(sums / counts.clamp(min=1.0), dim=-1)

                # --- I2P Term (Vectorized) ---
                # 原逻辑: sum(mean_l @ Pn[l].T) / n_classes
                # 向量化: (valid_means * Pn[valid_labels]).sum(dim=1).mean()
                # 解释: 两个归一化向量的点积等于元素对应相乘后求和
                selected_protos = Pn[unique_labels]
                # dot product per class
                sims_i2p = (class_means * selected_protos).sum(dim=1)
                i2p_term = sims_i2p.mean()

                if class_means.size(0) >= 2:
                    # [L_valid, L_valid]
                    cos_matrix = class_means @ class_means.t()
                    cos_matrix = cos_matrix.clamp(-1.0, 1.0)

                    inter = 1.0 - cos_matrix
                    inter.fill_diagonal_(0.0)
                    n_pairs = class_means.size(0) * (class_means.size(0) - 1)
                    inter_mean = inter.sum() / n_pairs
                else:
                    inter_mean = torch.tensor(0.0, device=m_u.device)

                loss = loss_ent - i2p_term - inter_mean

                self.model_tta_optimizer.zero_grad(set_to_none=True)

                # loss.backward()
                self.tta_scaler.scale(loss).backward()
                self.tta_scaler.unscale_(self.model_tta_optimizer)
                # 打印出梯度范数以供调试
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model_tta_params, max_norm=1.0
                )
                # logger.info(f"Model TTA Grad Norm: {norm:.4f}")
                norm = torch.nn.utils.clip_grad_norm_(self.model_tta_params, max_norm=1)
                # logger.info(f"Clipped Model TTA Grad Norm: {norm:.4f}")
                self.tta_scaler.step(self.model_tta_optimizer)
                self.tta_scaler.update()

        finally:
            if not was_training:
                self.model.eval()

    def _estimate_dynamic_tau(self, loader, protos, quantile):
        self.model.eval()
        device = self.device
        P = F.normalize(protos.to(device), dim=-1)
        sims_all = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                # FiveCrop: [B, 5, C, H, W] -> [B*5, C, H, W]
                if images.dim() == 5:
                    _, _, c, h, w = images.size()
                    images = images.view(-1, c, h, w)

                images = images.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda"):
                    feats = self.model(images).float()
                feats = F.normalize(feats, dim=-1)

                max_logit, _ = (feats @ P.T).max(dim=1)
                sims_all.append(max_logit.cpu())

        if len(sims_all) > 0:
            sims_all = torch.cat(sims_all)
            adaptive_tau = torch.quantile(sims_all, quantile).item()
            logger.info(
                f"Adaptive Tau: q={quantile}, tau={adaptive_tau:.4f} "
                f"(stats: min={sims_all.min():.4f}, mean={sims_all.mean():.4f})"
            )
            return adaptive_tau
        else:
            return 0.75

    def train_loop(
        self,
        dataset_train,
        dataset_test,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int = 8,
    ):
        def _train_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            y_true = np.asarray(y_true, dtype=np.int64)
            y_pred = np.asarray(y_pred, dtype=np.int64)
            n_samples = int(y_true.size)
            n_true = int(np.unique(y_true).size)
            n_pred = int(np.unique(y_pred).size)

            acc = cluster_acc(y_true, y_pred)

            # NMI (Normalized Mutual Information), pure-numpy implementation
            # NMI = MI / sqrt(H(true) * H(pred))
            true_ids, true_inv = np.unique(y_true, return_inverse=True)
            pred_ids, pred_inv = np.unique(y_pred, return_inverse=True)
            cont = np.zeros((true_ids.size, pred_ids.size), dtype=np.int64)
            np.add.at(cont, (true_inv, pred_inv), 1)
            n = cont.sum()
            if n == 0:
                nmi = 0.0
                ari = 0.0
            else:
                pi = cont.sum(axis=1).astype(np.float64)
                pj = cont.sum(axis=0).astype(np.float64)
                pij = cont.astype(np.float64)

                # mutual information
                nz = pij > 0
                mi = (pij[nz] / n) * np.log(
                    (pij[nz] * n) / (pi[:, None] * pj[None, :])[nz]
                )
                mi = float(mi.sum())

                # entropies
                p_true = pi / n
                p_pred = pj / n
                h_true = -float((p_true[p_true > 0] * np.log(p_true[p_true > 0])).sum())
                h_pred = -float((p_pred[p_pred > 0] * np.log(p_pred[p_pred > 0])).sum())
                denom = (h_true * h_pred) ** 0.5
                nmi = float(mi / denom) if denom > 0 else 0.0

                # ARI (Adjusted Rand Index), pure-numpy implementation
                def _comb2(x: np.ndarray) -> np.ndarray:
                    x = x.astype(np.float64)
                    return x * (x - 1.0) / 2.0

                sum_comb_c = float(_comb2(pij).sum())
                sum_comb_pi = float(_comb2(pi).sum())
                sum_comb_pj = float(_comb2(pj).sum())
                comb_n = float(n * (n - 1) / 2.0)
                expected = (sum_comb_pi * sum_comb_pj / comb_n) if comb_n > 0 else 0.0
                max_index = 0.5 * (sum_comb_pi + sum_comb_pj)
                denom_ari = max_index - expected
                ari = (
                    float((sum_comb_c - expected) / denom_ari) if denom_ari > 0 else 0.0
                )

            return {
                "acc": float(acc),
                "nmi": float(nmi),
                "ari": float(ari),
                "n_true": n_true,
                "n_pred": n_pred,
                "n_samples": n_samples,
            }

        orig_train_transform = dataset_train.transform
        orig_test_transform = dataset_test.transform
        contrast_train_transform = ContrastiveLearningViewGenerator(
            orig_train_transform, n_views=2
        )

        dataset_train.transform = contrast_train_transform

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        loader_test_unlabelled = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # 初始化原型设置为 NCM
        dataset_train.transform = orig_test_transform
        loader_train_raw = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=256,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        P_ncm = build_ncm_prototypes(
            loader=loader_train_raw,
            model=self.model,
            device=self.device,
            known_classes=self.train_classes,
        )
        dataset_train.transform = contrast_train_transform
        self.model.proto.data.copy_(P_ncm)

        for epoch in tqdm(
            range(self.start_epoch, self.epochs),
            desc="Overall Training Progress",
            ncols=130,
        ):
            self.train_one_epoch(
                epoch=epoch,
                loader_train=loader_train,
            )
            self.lr_scheduler.step(epoch + 1)

            # build NCM
            dataset_train.transform = orig_test_transform
            loader_train_raw = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=256,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            )
            P_ncm = build_ncm_prototypes(
                loader=loader_train_raw,
                model=self.model,
                device=self.device,
                known_classes=self.train_classes,
            )

            adaptive_tau = self._estimate_dynamic_tau(
                loader_train_raw, P_ncm, quantile=0.001
            )

            weight_back = self.model.save_visual_proj()
            if self.checkpoint_dir:
                checkpoint_path = f"{self.checkpoint_dir}/epoch_{epoch + 1:03d}_tau_{adaptive_tau:.4f}.pth"
                torch.save(
                    self.model.state_dict(),
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            # preds, targets, _, _, _ = self.evaluate(
            #     test_loader=loader_train_raw,
            #     tau=0,
            #     protos=P_ncm,
            #     tta_state=None,
            # )
            # m = _train_metrics(targets, preds)
            # logger.info(
            #     "Train Dataset Metrics Before Evaluation:\n"
            #     f"ACC={m['acc']:.4f}\n"
            #     f"NMI={m['nmi']:.4f}\n"
            #     f"ARI={m['ari']:.4f}\n"
            #     f"n_true={m['n_true']} n_pred={m['n_pred']} n_samples={m['n_samples']}"
            # )

            self.log_eval_stats(
                *self.evaluate(
                    test_loader=loader_test_unlabelled,
                    tau=self.tau,
                    protos=P_ncm,
                    tta_state=self.tta_state,
                )
            )

            # preds, targets, _, _, _ = self.evaluate(
            #     test_loader=loader_train_raw,
            #     tau=0,
            #     protos=P_ncm,
            #     tta_state=None,
            # )
            # m = _train_metrics(targets, preds)
            # logger.info(
            #     "Train Dataset Metrics After Evaluation:\n"
            #     f"ACC={m['acc']:.4f}\n"
            #     f"NMI={m['nmi']:.4f}\n"
            #     f"ARI={m['ari']:.4f}\n"
            #     f"n_true={m['n_true']} n_pred={m['n_pred']} n_samples={m['n_samples']}"
            # )

            dataset_train.transform = contrast_train_transform

            self.model.set_visual_proj(weight_back)

    def train_one_epoch(
        self,
        epoch: int,
        loader_train,
    ):
        self.model.train()
        device = self.device

        loss_ce_meter = SmoothedValue(window_size=20, fmt="{value:.4f}")
        loss_supcon_meter = SmoothedValue(window_size=20, fmt="{value:.4f}")
        lr_meter = SmoothedValue(window_size=1, fmt="{value:.6f}")

        logger.info(f"Start training epoch {epoch}")
        pbar = tqdm(loader_train, desc=f"Epoch [{epoch}]", ncols=130, leave=True)
        supcon = SupConLoss(temperature=self.supcon_t)

        arc_s = self.arc_s
        arc_m_now = self.arc_m

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                samples, targets = batch[0], batch[1]
            else:
                samples, targets = batch

            if isinstance(samples, list):
                x1 = samples[0].to(device, non_blocking=True)
                x2 = samples[1].to(device, non_blocking=True)
            else:
                x1 = x2 = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda"):
                f1 = self.model(x1, targets)
                f2 = self.model(x2, targets)
            f1 = f1.float()
            f2 = f2.float()

            phi, cos = angle_logits(f1, self.model.proto, s=arc_s, m=arc_m_now)
            logits = cos.clone()
            ar = torch.arange(logits.size(0), device=logits.device)
            logits[ar, targets] = phi[ar, targets]
            loss_ce = F.cross_entropy(logits, targets)

            feats = torch.stack([f1, f2], dim=1)
            loss_supcon = supcon(feats, labels=targets, device=str(device))

            loss = loss_supcon + loss_ce

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_ce_meter.update(float(loss_ce.detach()))
            loss_supcon_meter.update(float(loss_supcon.detach()))
            lr_meter.update(self.optimizer.param_groups[0]["lr"])

            pbar.set_postfix(
                {
                    "loss_ce": f"{loss_ce_meter.global_avg:.4f}",
                    "loss_supcon": f"{loss_supcon_meter.global_avg:.4f}",
                    "lr": f"{lr_meter.global_avg:.6f}",
                }
            )

        logger.info(
            f"Averaged stats - ce: {loss_ce_meter.global_avg:.4f} | "
            f"supcon: {loss_supcon_meter.global_avg:.4f} | "
            f"lr: {lr_meter.global_avg:.6f}"
        )

    def evaluate(
        self,
        test_loader,
        tau: float,
        protos: torch.Tensor,
        tta_state: Literal["M", "P", "M+P", None] = "M+P",
    ):
        self.model.eval()
        device = self.device

        P = F.normalize(protos.to(device), dim=-1)

        preds, targets, mask_old = [], [], []
        init_proto_count = int(P.size(0))
        new_class_count = 0

        K = len(self.train_classes)
        for images, label, *rest in tqdm(test_loader, desc="Eval"):
            images = images.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            targets.extend(label.tolist())
            mask_old.extend((label < K).tolist())

            p_ms = [[] for _ in range(P.size(0))]
            preds_batch = []
            # f_is = []
            # for i in range(images.size(0)):
            #     with torch.autocast(device_type="cuda"):
            #         f_i = self.model(images[i : i + 1])
            #     f_is.append(f_i.float())
            # f_is = torch.cat(f_is, dim=0)
            with torch.autocast(device_type="cuda"):
                f_is = self.model(images).float()
            f_is = f_is.float()
            f_is = F.normalize(f_is, dim=-1)
            for i in range(images.size(0)):
                f_i = f_is[i]
                logits_i = f_i @ P.T  # [1, P_curr]
                max_logit, pred = logits_i.max(dim=-1)
                pred_i = int(pred.item())

                if float(max_logit.item()) < tau:
                    new_vec = f_i.detach()
                    P = torch.cat([P, new_vec.unsqueeze(0)], dim=0)
                    new_class_count += 1
                    pred_i = P.size(0) - 1
                    p_ms.append([f_i.detach()])
                else:
                    pred_i = int(pred.item())
                    p_ms[pred_i].append(f_i.detach())

                preds.append(pred_i)
                preds_batch.append(pred_i)

            # model TTA
            if tta_state == "M" or tta_state == "M+P":
                _y_i = torch.tensor(preds_batch, device=device, dtype=torch.long)
                self.model_tta(images, _y_i, P)

            # # # proto TTA 自适应 START
            if tta_state == "P" or tta_state == "M+P":
                ema_new, ema_known = 0.3, 0.06
                smooth_k, smooth_k_known = 8.0, 32.0
                for j in range(K, len(p_ms)):
                    nj = len(p_ms[j])
                    if nj <= 1:
                        continue
                    # [n_j, D]
                    Xj = torch.stack(p_ms[j], dim=0)
                    sims = Xj @ P[j].detach()  # [n_j]
                    conf = float(sims.mean().clamp(0.0, 1.0))

                    mean_j = Xj.mean(dim=0)
                    alpha = ema_new * conf * (nj / (nj + smooth_k))
                    P[j] = F.normalize((1.0 - alpha) * P[j] + alpha * mean_j, dim=-1)

                for j in range(0, K):
                    nj = len(p_ms[j])
                    if nj <= 1:
                        continue

                    Xj = torch.stack(p_ms[j], dim=0)  # [n_j, D]
                    sims = Xj @ P[j].detach()  # [n_j]

                    mean_j = Xj.mean(dim=0)
                    conf = float(sims.mean().clamp(0.0, 1.0))
                    alpha = ema_known * conf * (nj / (nj + smooth_k_known))

                    P[j] = F.normalize((1.0 - alpha) * P[j] + alpha * mean_j, dim=-1)
            # proto tta 自适应 END

            P = F.normalize(P, dim=-1)

        return (
            np.array(preds, dtype=np.int64),
            np.array([int(x) for x in targets], dtype=np.int64),
            np.array(mask_old, dtype=bool),
            new_class_count,
            init_proto_count,
        )

    def log_eval_stats(
        self,
        preds,
        targets,
        mask_old,
        new_class_count,
        init_proto_count,
    ):
        all1, old1, new1 = split_cluster_acc_v1(
            y_true=targets, y_pred=preds, mask=mask_old
        )
        all2, old2, new2 = split_cluster_acc(
            y_true=targets, y_pred=preds, mask=mask_old
        )
        pretty = (
            f"[V1] all={all1:.3f} old={old1:.3f} new={new1:.3f}\n"
            f"[V2] all={all2:.3f} old={old2:.3f} new={new2:.3f}\n"
            f"[Grow] new_classes={new_class_count} (protos: {init_proto_count}→{init_proto_count + new_class_count})"
        )
        logger.info(f"Eval results::\n{pretty}")
