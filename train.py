import datetime
import os
import random
import time
from pathlib import Path
from typing import Literal, Optional

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
from tools.train_utils import get_best_gpu

metric_writer = None

DatasetName = Literal[
    "pets", "scars", "cub", "food", "imagenet100", "cifar10", "cifar100"
]


class Args(Tap):
    seed: int = 1028  # 随机种子
    dataset_name: DatasetName = "cub"  # 数据集
    backbone: Literal["clip", "dino"] = "clip"
    tta_state: Literal["M", "P", "M+P", None] = "M+P"
    device: str = ""
    tau: float = 0.75

    save_dir: str = "test"  # 模型和日志保存路径

    train_batch_size: int = 128
    eval_batch_size: int = 64
    num_workers: int = 8
    prop_train_labels: float = 0.5  # 训练集中有标签数据的比例

    start_epoch: int = 0
    epochs: int = 100

    clip_grad: Optional[float] = None

    # 以下数据不需要在命令行中指定
    train_classes: list[int] = []  # 训练时的类别
    unlabel_classes: list[int] = []  # 未标记数据的类别
    log_dir: str = ""
    checkpoint_dir: str = ""

    def process_args(self) -> None:
        if not self.device:
            best_gpu_id = get_best_gpu()
            self.device = f"cuda:{best_gpu_id}" if best_gpu_id is not None else "cpu"
        if self.save_dir:
            self.save_dir = os.path.join(
                "exp",
                self.save_dir,
                self.backbone,
                self.dataset_name,
            )
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = Path(self.save_dir, "checkpoints").as_posix()
            self.log_dir = Path(self.save_dir).as_posix()
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # fmt: off
        if self.dataset_name == "pets":
            self.train_classes = list(range(19))
            self.unlabel_classes = list(range(19, 37))
        elif self.dataset_name == "scars":
            self.train_classes = [1, 11, 25, 38, 46, 50, 53, 75, 84, 100, 105, 117, 123, 129, 133, 134, 135, 136, 137, 138, 140, 144, 145, 146, 147, 149, 150, 151, 153, 160, 161, 162, 163, 164, 167, 168, 169, 174, 175, 180, 185, 186, 187, 192, 193, 0, 81, 97, 104, 122, 139, 141, 142, 143, 148, 152, 154, 155, 156, 157, 158, 159, 165, 166, 170, 171, 172, 173, 176, 177, 181, 184, 188, 191, 194, 195, 2, 7, 9, 16, 20, 26, 28, 44, 54, 95, 98, 102, 127, 178, 182, 22, 41, 82, 93, 112, 125, 189]
            self.unlabel_classes = [23, 42, 83, 94, 113, 126, 190, 3, 8, 10, 17, 21, 27, 29, 45, 55, 96, 99, 103, 128, 179, 183, 4, 5, 6, 12, 13, 14, 15, 18, 19, 24, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 43, 47, 48, 49, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 85, 86, 87, 88, 89, 90, 91, 92, 101, 106, 107, 108, 109, 110, 111, 114, 115, 116, 118, 119, 120, 121, 124, 130, 131, 132]
        elif self.dataset_name == "cub":
            self.train_classes = [150, 70, 34, 178, 199, 131, 129, 147, 134, 11, 26, 93, 95, 121, 123, 99, 149, 167, 18, 31, 69, 198, 116, 158, 126, 17, 5, 179, 111, 163, 184, 81, 174, 42, 53, 89, 77, 55, 23, 48, 43, 44, 56, 28, 193, 143, 0, 176, 84, 15, 38, 154, 141, 190, 172, 124, 189, 19, 80, 157, 12, 9, 79, 30, 94, 67, 197, 97, 168, 137, 119, 76, 98, 88, 40, 106, 171, 87, 166, 186, 27, 51, 144, 135, 161, 64, 177, 7, 146, 61, 50, 162, 133, 82, 39, 74, 72, 91, 196, 136]
            self.unlabel_classes = [29, 110, 3, 8, 13, 58, 142, 25, 145, 63, 59, 65, 24, 140, 120, 32, 114, 107, 160, 130, 118, 101, 115, 128, 117, 71, 156, 112, 36, 122, 104, 102, 90, 125, 152, 195, 132, 83, 22, 192, 153, 175, 191, 155, 49, 194, 73, 66, 170, 151, 169, 96, 103, 37, 181, 127, 78, 21, 10, 164, 62, 2, 183, 85, 45, 60, 92, 185, 20, 159, 173, 148, 1, 57, 113, 165, 52, 109, 14, 4, 180, 6, 182, 68, 33, 108, 46, 35, 75, 188, 187, 100, 47, 105, 41, 86, 16, 54, 139, 138]
        elif self.dataset_name == "food":
            self.train_classes = list(range(51))
            self.unlabel_classes = list(range(51, 101))
        elif self.dataset_name == "imagenet100":
            self.train_classes = list(range(80))
            self.unlabel_classes = list(range(80, 100))
        elif self.dataset_name == "cifar10":
            self.train_classes = [0, 1, 2, 3, 4, 5]
            self.unlabel_classes = [6, 7, 8, 9]
        elif self.dataset_name == "cifar100":
            self.train_classes = list(range(80))
            self.unlabel_classes = list(range(80, 100))
        else:
            raise NotImplementedError
        # fmt: on


def get_outlog(args: Args):
    logger.remove()

    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

    log_name: str = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(Path(args.log_dir, log_name).as_posix(), level="INFO")

    global metric_writer
    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore


def main(args: Args):
    set_seed(args.seed)
    logger = get_outlog(args)
    args.save(
        Path(
            args.save_dir,
            f"args_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        ).as_posix()
    )
    device = torch.device(args.device)
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
        raise NotImplementedError
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabel_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]
    dataset_train.target_transform = target_transform
    dataset_test.target_transform = target_transform

    trainer = Trainer(
        tau=args.tau,
        tta_state=args.tta_state,
        epochs=args.epochs,
        train_classes=args.train_classes,
        unlabel_classes=args.unlabel_classes,
        device=device,
        start_epoch=args.start_epoch,
        max_norm=args.clip_grad,
        encoder=args.backbone,
        checkpoint_dir=args.checkpoint_dir,
    )

    n_parameters = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info("number of params: {}".format(n_parameters))

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    trainer.train_loop(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
