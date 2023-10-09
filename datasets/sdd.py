import os
from enum import Enum
import pickle

import cv2
import PIL
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SDDDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.split_id = int(classname)
        self.train_val_split = train_val_split
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize((int(resize*2.5+.5), resize)),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees, 
                                    translate=(translate, translate),
                                    scale=(1.0-scale, 1.0+scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((int(imagesize * 2.5 + .5), imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((int(resize*2.5+.5), resize)),
            transforms.CenterCrop((int(imagesize * 2.5 + .5), imagesize)),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, int(imagesize * 2.5 + .5), imagesize)
        
        # if self.split == DatasetSplit.TEST:
        #     for i in range(len(self.data_to_iterate)):
        #         self.__getitem__(i)

    def __getitem__(self, idx):
        data = self.data_to_iterate[idx]
        image = PIL.Image.open(data["img"]).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and data["anomaly"] == 1:
            mask = PIL.Image.open(data["label"])
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": str(self.split_id),
            "anomaly": data["anomaly"],
            "is_anomaly": data["anomaly"],
            "image_path": data["img"],
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):

        data_ids = []
        with open(os.path.join(self.source, "KolektorSDD-training-splits", "split.pyb"), "rb") as f:
            train_ids, test_ids, _ = pickle.load(f)
            if self.split == DatasetSplit.TRAIN:
                data_ids = train_ids[self.split_id]
            else:
                data_ids = test_ids[self.split_id]
        
        data = {}
        for data_id in data_ids:
            item_dir = os.path.join(self.source, data_id)
            fns = os.listdir(item_dir)
            part_ids = [os.path.splitext(fn)[0] for fn in fns if fn.endswith("jpg")]
            parts = {part_id:{"img":"", "label":"", "anomaly":0}
                     for part_id in part_ids}
            for part_id in parts:
                for fn in fns:
                    if part_id in fn:
                        if "label" in fn:
                            label = cv2.imread(os.path.join(item_dir, fn))
                            if label.sum() > 0:
                                parts[part_id]["anomaly"] = 1
                            parts[part_id]["label"] = os.path.join(item_dir, fn)
                        else:
                            parts[part_id]["img"] = os.path.join(item_dir, fn)
            for k, v in parts.items():
                if self.split == DatasetSplit.TRAIN and v["anomaly"] == 1:
                    continue
                data[data_id + '_' + k] = v

        return list(data.values())
