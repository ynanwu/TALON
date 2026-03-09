import os
from copy import deepcopy
from itertools import compress

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from scipy import io as mat_io
from torch.utils.data import Dataset
from torchvision.io import decode_image

cv2.setNumThreads(0)


class LabelRemapper:
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def __call__(self, label):
        return self.mapping_dict.get(label, label)


class CarsDataset(Dataset):
    def __init__(
        self, data_dir: str, train=True, transform=None, target_transform=None
    ):
        self.data_dir = os.path.join(data_dir, "cars_train/" if train else "cars_test/")
        meta_path = os.path.join(
            data_dir,
            "devkit/cars_train_annos.mat"
            if train
            else "devkit/cars_test_annos_withlabels.mat",
        )

        self.transform = transform
        self.target_transform = target_transform

        labels_meta = mat_io.loadmat(meta_path)

        self.samples = []
        self.targets = []

        for idx, img_ in enumerate(labels_meta["annotations"][0]):
            self.samples.append(os.path.join(self.data_dir, img_[5][0]))
            # Cars labels are 1-196, convert to 0-195
            self.targets.append(img_[4][0][0] - 1)

        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        path = self.samples[idx]
        target = self.targets[idx]
        uq_idx = self.uq_idxs[idx]

        img_tensor = decode_image(path)
        channels = int(img_tensor.shape[0])
        if channels == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif channels == 4:
            img_tensor = img_tensor[:3, ...]
        img = img_tensor.permute(1, 2, 0).numpy()

        if self.transform is not None:
            sample = self.transform(image=img)["image"]
        else:
            sample = torch.from_numpy(img).permute(2, 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, uq_idx


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset), dtype=bool)
    mask[idxs] = True

    dataset.samples = list(compress(dataset.samples, mask))
    dataset.targets = list(compress(dataset.targets, mask))
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.targets)
    train_idxs = []
    val_idxs = []

    for cls in train_classes:
        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]
        v_ = np.random.choice(
            cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),)
        )
        t_ = [x for x in cls_idxs if x not in v_]
        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_scars_datasets(
    root: str,
    train_transform,
    test_transform,
    train_classes: list = list(range(160)),
    prop_train_labels=0.8,
    seed=0,
):
    np.random.seed(seed)

    full_train_dataset = CarsDataset(
        data_dir=root, transform=train_transform, train=True
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


def get_scars_transform(
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
