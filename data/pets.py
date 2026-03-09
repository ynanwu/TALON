from copy import deepcopy
from itertools import compress

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.datasets import OxfordIIITPet
from torchvision.io import decode_image

cv2.setNumThreads(0)


# --- 1. 辅助类 ---
class LabelRemapper:
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def __call__(self, label):
        return self.mapping_dict.get(label, label)


class OxfordPetDataset(OxfordIIITPet):
    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform=None,
        target_transform=None,
        download: bool = True,
    ):
        """
        继承 torchvision 的 OxfordIIITPet 以利用其下载和解析逻辑。
        """
        super().__init__(
            root=root,
            split=split,
            transform=None,  # 禁用内置 transform，我们在 getitem 中处理
            target_transform=target_transform,
            download=download,
        )
        self.transform = transform
        self.samples = self._images
        self.targets = self._labels
        self.uq_idxs = np.array(np.arange(len(self.samples)), dtype=np.int32)

        print(f"Loaded {len(self.samples)} images from split '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:  # type: ignore
        path = self.samples[idx]
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


def subsample_dataset(dataset: OxfordPetDataset, idxs):
    mask = np.zeros(len(dataset), dtype=bool)
    mask[idxs] = True

    dataset.samples = list(compress(dataset.samples, mask))
    dataset.targets = list(compress(dataset.targets, mask))

    dataset._images = dataset.samples
    dataset._labels = dataset.targets

    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def get_oxford_datasets(
    root: str,
    train_transform,
    test_transform,
    train_classes: list[int],
    prop_train_labels: float,
    split: str = "trainval",
    seed=0,
):
    np.random.seed(seed)

    # 加载完整数据集 (作为 Pool)
    full_dataset = OxfordPetDataset(
        root=root,
        split=split,
        download=True,
    )

    all_targets = np.array(full_dataset.targets, dtype=np.int32)
    all_indices = np.array(full_dataset.uq_idxs, dtype=np.int32)

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
    train_dataset = subsample_dataset(deepcopy(full_dataset), train_indices)
    train_dataset.transform = train_transform
    # 标签重映射
    target_xform_dict = {k: i for i, k in enumerate(train_classes)}
    train_dataset.target_transform = LabelRemapper(target_xform_dict)

    # 构建 Test Dataset (Unlabelled / Validation)
    test_dataset = subsample_dataset(deepcopy(full_dataset), test_indices)
    test_dataset.transform = test_transform

    return train_dataset, test_dataset


def get_oxford_transform(
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
