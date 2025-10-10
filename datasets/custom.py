import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

import PIL
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CustomDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for SimpleNet custom folder datasets.

    The dataset expects an MVTec-like folder structure without the class
    hierarchy, e.g.:

    ```
    dataset_root/
        train/
            good/
                xxx.png
        test/
            good/
                xxx.png
            defect_type/
                xxx.png
        ground_truth/ (optional)
            defect_type/
                xxx.png
    ```

    The ``data_path`` argument passed from the command line can point to any of
    the directories inside this structure (for example, directly to
    ``dataset_root/train/good``); the loader will automatically walk up the
    directory tree until it finds the root folder that contains the ``train`` or
    ``test`` directory.
    """

    def __init__(
        self,
        source: str,
        classname: Optional[str] = None,
        resize: int = 256,
        imagesize: int = 224,
        split: DatasetSplit = DatasetSplit.TRAIN,
        train_val_split: float = 1.0,
        rotate_degrees: int = 0,
        translate: float = 0.0,
        brightness_factor: float = 0.0,
        contrast_factor: float = 0.0,
        saturation_factor: float = 0.0,
        gray_p: float = 0.0,
        h_flip_p: float = 0.0,
        v_flip_p: float = 0.0,
        scale: float = 0.0,
        augment: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.source = os.path.abspath(source)
        self.split = split
        self.train_val_split = train_val_split

        self.dataset_root = self._find_dataset_root(self.source)
        self.classname = classname or os.path.basename(self.dataset_root.rstrip(os.sep))
        if not self.classname:
            self.classname = "custom"

        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

        interpolation = transforms.InterpolationMode.BILINEAR
        self.transform_img = [
            transforms.Resize(resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor)
            if augment
            else transforms.Lambda(lambda img: img),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(
                rotate_degrees,
                translate=(translate, translate),
                scale=(1.0 - scale, 1.0 + scale),
                interpolation=interpolation,
            ),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        # Remove the identity transform when augmentation is disabled to avoid
        # torchscript complaints about lambdas later on.
        if not augment:
            self.transform_img.pop(1)
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )

        self.imagesize = (3, imagesize, imagesize)

        (
            self.imgpaths_per_class,
            self.maskpaths_per_class,
            self.data_to_iterate,
        ) = self.get_image_data()

    def __len__(self) -> int:
        return len(self.data_to_iterate)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        image_name = os.path.relpath(image_path, self.dataset_root)

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": image_name,
            "image_path": image_path,
        }

    def _find_dataset_root(self, start_path: str) -> str:
        current = os.path.abspath(start_path)
        while True:
            train_dir = os.path.join(current, "train")
            test_dir = os.path.join(current, "test")
            if os.path.isdir(train_dir) or os.path.isdir(test_dir):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        raise FileNotFoundError(
            f"Could not locate a dataset root starting from '{start_path}'. "
            "Expected a directory containing a 'train' or 'test' folder."
        )

    def _gather_split_directories(self) -> Tuple[str, List[str]]:
        split_dir = os.path.join(self.dataset_root, self.split.value)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory '{split_dir}' does not exist for dataset '{self.dataset_root}'."
            )

        subdirs = [
            d
            for d in sorted(os.listdir(split_dir))
            if os.path.isdir(os.path.join(split_dir, d))
        ]

        if not subdirs:
            # Allow datasets where images are stored directly in the split
            # directory without an extra anomaly-type subfolder.
            subdirs = ["good"]

        return split_dir, subdirs

    def _gather_images(self, folder: str) -> List[str]:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return [
            os.path.join(folder, file_name)
            for file_name in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, file_name))
            and os.path.splitext(file_name)[1].lower() in valid_extensions
        ]

    def get_image_data(
        self,
    ) -> Tuple[
        Dict[str, Dict[str, List[str]]],
        Dict[str, Dict[str, Optional[List[str]]]],
        List[Tuple[str, str, str, Optional[str]]],
    ]:
        split_dir, anomaly_types = self._gather_split_directories()

        mask_root_candidates = [
            os.path.join(self.dataset_root, "ground_truth"),
            os.path.join(self.dataset_root, "groundtruth"),
        ]
        mask_root = next((p for p in mask_root_candidates if os.path.isdir(p)), None)

        imgpaths_per_class: Dict[str, Dict[str, List[str]]] = {self.classname: {}}
        maskpaths_per_class: Dict[str, Dict[str, Optional[List[str]]]] = {self.classname: {}}
        data_to_iterate: List[Tuple[str, str, str, Optional[str]]] = []

        for anomaly in anomaly_types:
            if anomaly == "good" and anomaly != anomaly_types[0]:
                anomaly_folder = os.path.join(split_dir, anomaly)
            elif anomaly_types == ["good"]:
                anomaly_folder = split_dir
            else:
                anomaly_folder = os.path.join(split_dir, anomaly)

            image_paths = self._gather_images(anomaly_folder)
            if not image_paths:
                continue

            imgpaths_per_class[self.classname][anomaly] = image_paths

            if self.train_val_split < 1.0:
                n_images = len(image_paths)
                split_idx = int(n_images * self.train_val_split)
                if self.split == DatasetSplit.TRAIN:
                    imgpaths_per_class[self.classname][anomaly] = image_paths[:split_idx]
                elif self.split == DatasetSplit.VAL:
                    imgpaths_per_class[self.classname][anomaly] = image_paths[split_idx:]

            if self.split == DatasetSplit.TEST and anomaly != "good" and mask_root is not None:
                anomaly_mask_folder = os.path.join(mask_root, anomaly)
                if os.path.isdir(anomaly_mask_folder):
                    mask_files = self._gather_images(anomaly_mask_folder)
                else:
                    mask_files = []
                maskpaths_per_class[self.classname][anomaly] = mask_files if mask_files else None
            else:
                maskpaths_per_class[self.classname][anomaly] = None

            for i, image_path in enumerate(imgpaths_per_class[self.classname][anomaly]):
                if self.split == DatasetSplit.TEST and anomaly != "good":
                    mask_list = maskpaths_per_class[self.classname].get(anomaly)
                    mask_path = mask_list[i] if mask_list and len(mask_list) > i else None
                else:
                    mask_path = None
                data_to_iterate.append(
                    (self.classname, anomaly, image_path, mask_path)
                )

        if not data_to_iterate:
            raise RuntimeError(
                f"No images found in split '{self.split.value}' for dataset rooted at '{self.dataset_root}'."
            )

        return imgpaths_per_class, maskpaths_per_class, data_to_iterate
