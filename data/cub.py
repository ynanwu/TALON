import os
from copy import deepcopy
from itertools import compress

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.io import decode_image

cv2.setNumThreads(0)


# --- 1. 辅助类 ---
class LabelRemapper:
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def __call__(self, label):
        return self.mapping_dict.get(label, label)


class Cub2011Dataset(Dataset):
    base_folder = "CUB_200_2011/images"
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: str,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id").merge(
            train_test_split, on="img_id"
        )

        if self.train:
            data = data[data.is_training_img == 1]
        else:
            data = data[data.is_training_img == 0]
        self.samples = [
            os.path.join(self.root, self.base_folder, path)
            for path in data["filepath"].values.tolist()
        ]
        self.targets = (data["target"] - 1).values.tolist()

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for filepath in self.samples:
            filepath = os.path.join(self.root, self.base_folder, filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

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

    dataset.samples = list(compress(dataset.samples, mask))
    dataset.targets = list(compress(dataset.targets, mask))
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def get_cub_datasets(
    root: str,
    train_transform,
    test_transform,
    train_classes: list[int],
    prop_train_labels: float,
    seed=0,
):
    np.random.seed(seed)

    # 加载完整数据集 (作为 Pool)
    full_train_dataset = Cub2011Dataset(
        root=root,
        train=True,
        download=True,
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


def get_cub_transform(
    seed: int,
    interpolation=cv2.INTER_CUBIC,
    crop_pct=0.875,
    image_size=224,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
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
