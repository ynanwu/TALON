import os
from copy import deepcopy
from itertools import compress

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torchvision.io import decode_image

cv2.setNumThreads(0)


# --- 1. 辅助类 ---
class LabelRemapper:
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def __call__(self, label):
        return self.mapping_dict.get(label, label)


class ImageNetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform=None)

        self.uq_idxs = np.array(range(len(self)))
        self.paths = [sample[0] for sample in self.samples]
        self.targets = [sample[1] for sample in self.samples]

    def __getitem__(self, idx: int):  # type: ignore
        path = self.paths[idx]
        target = self.targets[idx]
        uq_idx = self.uq_idxs[idx]

        img_tensor = decode_image(str(path))
        # img = cv2.imread(path)
        # cv2.imwrite(path, img)

        channels = int(img_tensor.shape[0])
        if channels == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif channels == 4:
            img_tensor = img_tensor[:3, ...]
        img = img_tensor.permute(1, 2, 0).numpy()  # (H, W, C)

        if self.transform is not None:
            sample: torch.Tensor = self.transform(image=img)["image"]
        else:
            sample = torch.from_numpy(img).permute(2, 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, uq_idx


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset), dtype=bool)
    mask[idxs] = True

    # ImageNetDataset 继承自 ImageFolder，__len__ 依赖 dataset.samples。
    # 因此这里必须同步更新 samples/imgs，避免 len(dataset) 与 paths/targets 不一致。
    dataset.samples = list(compress(dataset.samples, mask))
    dataset.imgs = dataset.samples

    dataset.paths = [sample[0] for sample in dataset.samples]
    dataset.targets = [sample[1] for sample in dataset.samples]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def get_imagenet_100_datasets(
    root: str,
    train_transform,
    test_transform,
    train_classes: list[int],
    prop_train_labels: float,
    seed=0,
):
    np.random.seed(seed)

    imagenet_full_train = ImageNetDataset(
        root=os.path.join(root, "train"), transform=None
    )
    all_targets_full = np.array(imagenet_full_train.targets, dtype=np.int32)
    all_indices_full = np.arange(len(imagenet_full_train), dtype=np.int32)

    num_classes_in_root = len(imagenet_full_train.classes)
    if num_classes_in_root == 1000:
        subsampled_100_classes = np.random.choice(
            range(1000), size=(100,), replace=False
        )
        subsampled_100_classes = np.sort(subsampled_100_classes)
        print(
            f"Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}"
        )

        in_100_mask = np.isin(all_targets_full, subsampled_100_classes)
        imagenet100_indices = all_indices_full[in_100_mask]
        full_dataset = subsample_dataset(
            deepcopy(imagenet_full_train), imagenet100_indices
        )

        cls_map = {
            old_cls: new_cls for new_cls, old_cls in enumerate(subsampled_100_classes)
        }
        full_dataset.targets = [cls_map[t] for t in full_dataset.targets]
        full_dataset.samples = [(p, cls_map[t]) for (p, t) in full_dataset.samples]
    elif num_classes_in_root == 100:
        print("Detected 100-class ImageNet root, using all classes directly.")
        full_dataset = deepcopy(imagenet_full_train)
    else:
        raise ValueError(
            f"Unsupported ImageNet root with {num_classes_in_root} classes. "
            "Expected either 1000 (ImageNet-1K) or 100 (ImageNet-100)."
        )

    # 重置 uq_idxs 为 0..N-1（保持与 pets/cub 的 subsample 逻辑一致）
    full_dataset.imgs = full_dataset.samples
    full_dataset.uq_idxs = np.arange(len(full_dataset), dtype=np.int32)
    full_dataset.transform = None
    full_dataset.target_transform = None

    all_targets = np.array(full_dataset.targets, dtype=np.int32)
    all_indices = np.array(full_dataset.uq_idxs, dtype=np.int32)

    # 逻辑：Known 类 (Train Classes) vs Unknown 类
    known_mask = np.isin(all_targets, train_classes)
    known_indices = all_indices[known_mask]
    unknown_indices = all_indices[~known_mask]

    if len(unknown_indices) == 0:
        raise ValueError(
            "No unknown-class samples found in test split. "
            f"train_classes={train_classes}, unique_targets={np.unique(all_targets).tolist()}"
        )
    if len(known_indices) == 0:
        raise ValueError(
            "No known-class samples found. Please check train_classes and dataset labels."
        )

    # Shuffle Known 数据
    np.random.shuffle(known_indices)

    # 划分 Labelled (Train)
    split_point = int(len(known_indices) * prop_train_labels)
    train_indices = known_indices[:split_point]

    # 划分 Unlabelled/Test (剩余的 Known + 所有的 Unknown)
    test_indices = np.concatenate([known_indices[split_point:], unknown_indices])

    print(f"Splitting: Train={len(train_indices)}, Test={len(test_indices)}")

    # 构建 Train Dataset (Labelled)
    train_dataset = subsample_dataset(deepcopy(full_dataset), train_indices)
    train_dataset.transform = train_transform
    target_xform_dict = {k: i for i, k in enumerate(train_classes)}
    train_dataset.target_transform = LabelRemapper(target_xform_dict)

    # 构建 Test Dataset (Unlabelled / Validation)
    test_dataset = subsample_dataset(deepcopy(full_dataset), test_indices)
    test_dataset.transform = test_transform

    return train_dataset, test_dataset


def get_imagenet_100_transform(
    seed: int,
    interpolation=cv2.INTER_CUBIC,
    crop_pct=0.875,
    image_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    resize_size = int(image_size / crop_pct)

    train_transform = A.Compose(
        [
            A.Resize(resize_size, resize_size, interpolation=interpolation),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.8),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        seed=seed,
    )
    test_transform = A.Compose(
        [
            A.Resize(resize_size, resize_size, interpolation=interpolation),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        seed=seed,
    )
    return (train_transform, test_transform)
