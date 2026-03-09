from copy import deepcopy

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.datasets import CIFAR10, CIFAR100

cv2.setNumThreads(0)


class LabelRemapper:
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def __call__(self, label):
        return self.mapping_dict.get(label, label)


class CIFAR10Dataset(CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10Dataset, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):  # type: ignore
        img, target = self.data[item], self.targets[item]
        uq_idx = self.uq_idxs[item]

        img_tensor = torch.tensor(img)
        img = img_tensor.numpy()  # (H, W, C)

        channels = int(img_tensor.shape[0])
        if channels == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif channels == 4:
            img_tensor = img_tensor[:3, ...]

        if self.transform is not None:
            sample: torch.Tensor = self.transform(image=img)["image"]
        else:
            sample = torch.from_numpy(img).permute(2, 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, uq_idx

    def __len__(self):
        return len(self.targets)


class CIFAR100Dataset(CIFAR100):
    def __init__(self, *args, **kwargs):
        super(CIFAR100Dataset, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):  # type: ignore
        img, target = self.data[item], self.targets[item]
        uq_idx = self.uq_idxs[item]

        img_tensor = torch.tensor(img)
        img = img_tensor.numpy()  # (H, W, C)

        channels = int(img_tensor.shape[0])
        if channels == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif channels == 4:
            img_tensor = img_tensor[:3, ...]

        if self.transform is not None:
            sample: torch.Tensor = self.transform(image=img)["image"]
        else:
            sample = torch.from_numpy(img).permute(2, 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):
    dataset.data = dataset.data[idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def get_cifar_10_datasets(
    root: str,
    train_transform,
    test_transform,
    train_classes: list[int],
    prop_train_labels: float,
    seed=0,
):
    np.random.seed(seed)

    full_train_dataset = CIFAR10Dataset(
        root=root, transform=None, train=True, download=True
    )

    all_targets = np.array(full_train_dataset.targets, dtype=np.int32)
    all_indices = np.array(full_train_dataset.uq_idxs, dtype=np.int32)
    # 逻辑：Known 类 (Train Classes) vs Unknown 类
    known_mask = np.isin(all_targets, train_classes)
    known_indices = all_indices[known_mask]
    unknown_indices = all_indices[~known_mask]

    # Shuffle Known 数据
    np.random.shuffle(known_indices)

    # 划分 Labelled (Train)
    split_point = int(len(known_indices) * prop_train_labels)
    train_indices = known_indices[:split_point]

    # 划分 Unlabelled/Test (剩余的 Known + 所有的 Unknown)
    test_indices = np.concatenate([known_indices[split_point:], unknown_indices])

    print(f"Splitting: Train={len(train_indices)}, Test={len(test_indices)}")

    # 构建 Train Dataset (Labelled)
    train_dataset = subsample_dataset(deepcopy(full_train_dataset), train_indices)
    train_dataset.transform = train_transform
    # 标签重映射
    target_xform_dict = {k: i for i, k in enumerate(train_classes)}
    train_dataset.target_transform = LabelRemapper(target_xform_dict)

    # 构建 Test Dataset (Unlabelled / Validation)
    test_dataset = subsample_dataset(deepcopy(full_train_dataset), test_indices)
    test_dataset.transform = test_transform

    return train_dataset, test_dataset


def get_cifar_100_datasets(
    root: str,
    train_transform,
    test_transform,
    train_classes: list[int],
    prop_train_labels: float,
    seed=0,
):
    np.random.seed(seed)

    full_train_dataset = CIFAR100Dataset(
        root=root, transform=train_transform, train=True, download=True
    )

    all_targets = np.array(full_train_dataset.targets, dtype=np.int32)
    all_indices = np.array(full_train_dataset.uq_idxs, dtype=np.int32)

    # 逻辑：Known 类 (Train Classes) vs Unknown 类
    known_mask = np.isin(all_targets, train_classes)
    known_indices = all_indices[known_mask]
    unknown_indices = all_indices[~known_mask]

    # Shuffle Known 数据
    np.random.shuffle(known_indices)

    # 划分 Labelled (Train)
    split_point = int(len(known_indices) * prop_train_labels)
    train_indices = known_indices[:split_point]

    # 划分 Unlabelled/Test (剩余的 Known + 所有的 Unknown)
    test_indices = np.concatenate([known_indices[split_point:], unknown_indices])

    print(f"Splitting: Train={len(train_indices)}, Test={len(test_indices)}")

    # 构建 Train Dataset (Labelled)
    train_dataset = subsample_dataset(deepcopy(full_train_dataset), train_indices)
    train_dataset.transform = train_transform
    # 标签重映射
    target_xform_dict = {k: i for i, k in enumerate(train_classes)}
    train_dataset.target_transform = LabelRemapper(target_xform_dict)

    # 构建 Test Dataset (Unlabelled / Validation)
    test_dataset = subsample_dataset(deepcopy(full_train_dataset), test_indices)
    test_dataset.transform = test_transform

    return train_dataset, test_dataset


def get_cifar_transform(
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
