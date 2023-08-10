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


class SDD2Dataset(torch.utils.data.Dataset):
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
        img_path, gt_path, is_anomaly = self.data_to_iterate[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and is_anomaly:
            mask = PIL.Image.open(gt_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": "",
            "anomaly": is_anomaly,
            "is_anomaly": is_anomaly,
            "image_path": img_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):

        data_ids = []
        
        data_dir = os.path.join(self.source, "train" if self.split == DatasetSplit.TRAIN else "test")
        data = []
        test = [0, 0]
        for fn in os.listdir(data_dir):
            if "GT" not in fn:
                data_id = os.path.splitext(fn)[0]
                img_path = os.path.join(data_dir, fn)
                gt_path = os.path.join(data_dir, f"{data_id}_GT.png")
                assert os.path.exists(img_path)
                assert os.path.exists(gt_path), gt_path
                gt = cv2.imread(gt_path)
                is_anomaly = gt.sum() > 0
                if is_anomaly:
                    test[1] = test[1] + 1
                else:
                    test[0] = test[0] + 1
                if self.split == DatasetSplit.TRAIN and is_anomaly:
                    continue
                data.append([img_path, gt_path, gt.sum() > 0])

        return data
