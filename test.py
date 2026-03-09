import datetime
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from loguru import logger
from tap import Tap
from tqdm import tqdm

from config import (
    CIFAR_10_ROOT,
    CIFAR_100_ROOT,
    CUB_ROOT,
    FOOD_101_ROOT,
    IMAGENET_ROOT,
    OXFORD_PET_ROOT,
    SCARS_ROOT,
)
from data.pets import get_oxford_datasets, get_oxford_transform
from methods.talon.trainer import Trainer
from methods.talon.utils import build_ncm_prototypes, split_cluster_acc_v1
from tools.evaluate_utils import split_cluster_acc
from tools.train_utils import get_best_gpu

DatasetName = Literal[
    "pets", "scars", "cub", "food", "imagenet100", "cifar10", "cifar100"
]

BACKBONE_SET = {"clip", "dino"}
DATASET_SET = {"pets", "scars", "cub", "food", "imagenet100", "cifar10", "cifar100"}


class Args(Tap):
    ckpt_path: str

    seed: int = 1028
    dataset_name: DatasetName = "cub"
    backbone: Literal["clip", "dino"] = "clip"
    tau: float = 0.75

    device: str = ""
    tta_state: Literal["M", "P", "M+P", "none"] = "M+P"

    save_dir: str = "test_eval"
    eval_batch_size: int = 64
    num_workers: int = 8
    prop_train_labels: float = 0.5

    train_classes: list[int] = []
    unlabel_classes: list[int] = []
    log_dir: str = ""

    def process_args(self) -> None:
        if not self.device:
            best_gpu_id = get_best_gpu()
            self.device = f"cuda:{best_gpu_id}" if best_gpu_id is not None else "cpu"
        checkpoint = Path(self.ckpt_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        self.ckpt_path = checkpoint.as_posix()

        if self.save_dir:
            self.save_dir = os.path.join(
                "exp",
                self.save_dir,
                self.backbone,
                self.dataset_name,
            )
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            self.log_dir = Path(self.save_dir).as_posix()

        assign_class_splits(self)


def assign_class_splits(args: Args) -> None:
    # fmt: off
    if args.dataset_name == "pets":
        args.train_classes = list(range(19))
        args.unlabel_classes = list(range(19, 37))
    elif args.dataset_name == "scars":
        args.train_classes = [1, 11, 25, 38, 46, 50, 53, 75, 84, 100, 105, 117, 123, 129, 133, 134, 135, 136, 137, 138, 140, 144, 145, 146, 147, 149, 150, 151, 153, 160, 161, 162, 163, 164, 167, 168, 169, 174, 175, 180, 185, 186, 187, 192, 193, 0, 81, 97, 104, 122, 139, 141, 142, 143, 148, 152, 154, 155, 156, 157, 158, 159, 165, 166, 170, 171, 172, 173, 176, 177, 181, 184, 188, 191, 194, 195, 2, 7, 9, 16, 20, 26, 28, 44, 54, 95, 98, 102, 127, 178, 182, 22, 41, 82, 93, 112, 125, 189]
        args.unlabel_classes = [23, 42, 83, 94, 113, 126, 190, 3, 8, 10, 17, 21, 27, 29, 45, 55, 96, 99, 103, 128, 179, 183, 4, 5, 6, 12, 13, 14, 15, 18, 19, 24, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 43, 47, 48, 49, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 85, 86, 87, 88, 89, 90, 91, 92, 101, 106, 107, 108, 109, 110, 111, 114, 115, 116, 118, 119, 120, 121, 124, 130, 131, 132]
    elif args.dataset_name == "cub":
        args.train_classes = [150, 70, 34, 178, 199, 131, 129, 147, 134, 11, 26, 93, 95, 121, 123, 99, 149, 167, 18, 31, 69, 198, 116, 158, 126, 17, 5, 179, 111, 163, 184, 81, 174, 42, 53, 89, 77, 55, 23, 48, 43, 44, 56, 28, 193, 143, 0, 176, 84, 15, 38, 154, 141, 190, 172, 124, 189, 19, 80, 157, 12, 9, 79, 30, 94, 67, 197, 97, 168, 137, 119, 76, 98, 88, 40, 106, 171, 87, 166, 186, 27, 51, 144, 135, 161, 64, 177, 7, 146, 61, 50, 162, 133, 82, 39, 74, 72, 91, 196, 136]
        args.unlabel_classes = [29, 110, 3, 8, 13, 58, 142, 25, 145, 63, 59, 65, 24, 140, 120, 32, 114, 107, 160, 130, 118, 101, 115, 128, 117, 71, 156, 112, 36, 122, 104, 102, 90, 125, 152, 195, 132, 83, 22, 192, 153, 175, 191, 155, 49, 194, 73, 66, 170, 151, 169, 96, 103, 37, 181, 127, 78, 21, 10, 164, 62, 2, 183, 85, 45, 60, 92, 185, 20, 159, 173, 148, 1, 57, 113, 165, 52, 109, 14, 4, 180, 6, 182, 68, 33, 108, 46, 35, 75, 188, 187, 100, 47, 105, 41, 86, 16, 54, 139, 138]
    elif args.dataset_name == "food":
        args.train_classes = list(range(51))
        args.unlabel_classes = list(range(51, 101))
    elif args.dataset_name == "imagenet100":
        args.train_classes = list(range(80))
        args.unlabel_classes = list(range(80, 100))
    elif args.dataset_name == "cifar10":
        args.train_classes = [0, 1, 2, 3, 4, 5]
        args.unlabel_classes = [6, 7, 8, 9]
    elif args.dataset_name == "cifar100":
        args.train_classes = list(range(80))
        args.unlabel_classes = list(range(80, 100))
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset_name}")
    # fmt: on


def get_outlog(args: Args):
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

    log_name = f"test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(Path(args.log_dir, log_name).as_posix(), level="INFO")
    return logger


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore


def build_datasets(args: Args):
    if args.backbone == "dino":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

    if args.dataset_name == "pets":
        train_transform, test_transform = get_oxford_transform(
            args.seed, mean=mean, std=std
        )
        dataset_train, dataset_test = get_oxford_datasets(
            root=OXFORD_PET_ROOT,
            train_transform=train_transform,
            test_transform=test_transform,
            train_classes=args.train_classes,
            prop_train_labels=args.prop_train_labels,
            seed=args.seed,
        )
    elif args.dataset_name == "scars":
        from data.scars import get_scars_datasets, get_scars_transform

        train_transform, test_transform = get_scars_transform(
            args.seed, mean=mean, std=std
        )
        dataset_train, dataset_test = get_scars_datasets(
            root=SCARS_ROOT,
            train_transform=train_transform,
            test_transform=test_transform,
            train_classes=args.train_classes,
            prop_train_labels=args.prop_train_labels,
            seed=args.seed,
        )
    elif args.dataset_name == "cub":
        from data.cub import get_cub_datasets, get_cub_transform

        train_transform, test_transform = get_cub_transform(
            args.seed, mean=mean, std=std
        )
        dataset_train, dataset_test = get_cub_datasets(
            root=CUB_ROOT,
            train_transform=train_transform,
            test_transform=test_transform,
            train_classes=args.train_classes,
            prop_train_labels=args.prop_train_labels,
            seed=args.seed,
        )
    elif args.dataset_name == "cifar10":
        from data.cifar import get_cifar_10_datasets, get_cifar_transform

        train_transform, test_transform = get_cifar_transform(
            args.seed, mean=mean, std=std
        )
        dataset_train, dataset_test = get_cifar_10_datasets(
            root=CIFAR_10_ROOT,
            train_transform=train_transform,
            test_transform=test_transform,
            train_classes=args.train_classes,
            prop_train_labels=args.prop_train_labels,
            seed=args.seed,
        )
    elif args.dataset_name == "cifar100":
        from data.cifar import get_cifar_100_datasets, get_cifar_transform

        train_transform, test_transform = get_cifar_transform(
            args.seed, mean=mean, std=std
        )
        dataset_train, dataset_test = get_cifar_100_datasets(
            root=CIFAR_100_ROOT,
            train_transform=train_transform,
            test_transform=test_transform,
            train_classes=args.train_classes,
            prop_train_labels=args.prop_train_labels,
            seed=args.seed,
        )
    elif args.dataset_name == "food":
        from data.food101 import get_food101_transform, get_food_101_datasets

        train_transform, test_transform = get_food101_transform(
            args.seed, mean=mean, std=std
        )
        dataset_train, dataset_test = get_food_101_datasets(
            root=FOOD_101_ROOT,
            train_transform=train_transform,
            test_transform=test_transform,
            train_classes=args.train_classes,
            prop_train_labels=args.prop_train_labels,
            seed=args.seed,
        )
    elif args.dataset_name == "imagenet100":
        from data.imagenet import get_imagenet_100_datasets, get_imagenet_100_transform

        train_transform, test_transform = get_imagenet_100_transform(
            args.seed, mean=mean, std=std
        )
        dataset_train, dataset_test = get_imagenet_100_datasets(
            root=IMAGENET_ROOT,
            train_transform=train_transform,
            test_transform=test_transform,
            train_classes=args.train_classes,
            prop_train_labels=args.prop_train_labels,
            seed=args.seed,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset_name}")

    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabel_classes)):
        target_transform_dict[cls] = i

    def target_transform(x):
        return target_transform_dict[x]

    dataset_train.target_transform = target_transform
    dataset_test.target_transform = target_transform

    dataset_train.transform = test_transform
    dataset_test.transform = test_transform
    return dataset_train, dataset_test


def eval_one_split(
    split_name: str,
    trainer: Trainer,
    dataset,
    protos: torch.Tensor,
    batch_size: int,
    num_workers: int,
    tau: float,
    tta_state: Literal["M", "P", "M+P", None],
):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    preds, targets, mask_old, new_class_count, init_proto_count = trainer.evaluate(
        test_loader=loader,
        tau=tau,
        protos=protos.clone(),
        tta_state=tta_state,
    )
    all1, old1, new1 = split_cluster_acc_v1(
        y_true=targets,
        y_pred=preds,
        mask=mask_old,
    )
    all2, old2, new2 = split_cluster_acc(
        y_true=targets,
        y_pred=preds,
        mask=mask_old,
    )

    logger.info(
        f"[{split_name}] tau={tau:.4f} tta={tta_state}\n"
        f"[V1] all={all1:.4f} old={old1:.4f} new={new1:.4f}\n"
        f"[V2] all={all2:.4f} old={old2:.4f} new={new2:.4f}\n"
        f"[Grow] new_classes={new_class_count} (protos: {init_proto_count}->{init_proto_count + new_class_count})"
    )

    return {
        "split": split_name,
        "tau": float(tau),
        "tta_state": tta_state,
        "v1": {"all": float(all1), "old": float(old1), "new": float(new1)},
        "v2": {"all": float(all2), "old": float(old2), "new": float(new2)},
        "grow": {
            "new_classes": int(new_class_count),
            "init_proto_count": int(init_proto_count),
            "final_proto_count": int(init_proto_count + new_class_count),
        },
        "num_samples": int(len(targets)),
    }


def main(args: Args):
    set_seed(args.seed)
    get_outlog(args)

    if args.tau is None:
        raise ValueError("tau must be resolved before evaluation")

    logger.info(f"checkpoint: {args.ckpt_path}")
    logger.info(
        f"dataset={args.dataset_name} backbone={args.backbone} device={args.device} "
        f"tau={float(args.tau):.4f} tta={args.tta_state}"
    )

    device = torch.device(args.device)
    dataset_train, dataset_test = build_datasets(args)
    checkpoint_state = torch.load(args.ckpt_path, map_location="cpu")

    trainer = Trainer(
        tau=float(args.tau),
        epochs=1,
        train_classes=args.train_classes,
        unlabel_classes=args.unlabel_classes,
        device=device,
        start_epoch=0,
        max_norm=None,
        encoder=args.backbone,
        checkpoint_dir=None,
    )
    load_result = trainer.model.load_state_dict(checkpoint_state, strict=False)
    if load_result.missing_keys:
        logger.warning(
            f"Missing keys when loading checkpoint: {load_result.missing_keys}"
        )
    if load_result.unexpected_keys:
        logger.warning(
            f"Unexpected keys when loading checkpoint: {load_result.unexpected_keys}"
        )

    train_loader_for_proto = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    P_ncm = build_ncm_prototypes(
        loader=train_loader_for_proto,
        model=trainer.model,
        device=device,
        known_classes=args.train_classes,
    )
    test_tta_state = None if args.tta_state == "none" else args.tta_state
    test_metrics = eval_one_split(
        split_name="test",
        trainer=trainer,
        dataset=dataset_test,
        protos=P_ncm,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        tau=float(args.tau),
        tta_state=test_tta_state,
    )

    summary_path = Path(
        args.save_dir,
        f"eval_{Path(args.ckpt_path).stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    summary = [
        f"checkpoint: {args.ckpt_path}",
        f"dataset: {args.dataset_name}",
        f"backbone: {args.backbone}",
        f"device: {args.device}",
        f"seed: {args.seed}",
        f"prop_train_labels: {args.prop_train_labels}",
        f"tau: {args.tau}",
        "proto_source: P_ncm",
        f"test_tta_state: {test_tta_state}",
        "",
        f"test v1: {test_metrics['v1']}",
        f"test v2: {test_metrics['v2']}",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    logger.info(f"Saved summary to {summary_path.as_posix()}")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
