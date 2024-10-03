import os

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms


class MstarGenerationDataset(Dataset):
    name_to_type_idx = {
        '2S1': 0,
        'BMP2': 1,
        'BRDM2': 2,
        'BTR60': 3,
        'BTR70': 4,
        'D7': 5,
        'T62': 6,
        'T72': 7,
        'ZIL131': 8,
        'ZSU23_4': 9,
    }

    type_idx_to_name = {v: k for k, v in name_to_type_idx.items()}

    def __init__(self, base_dir, is_val=False, img_transform=None, use_pil=False, aug_ratio=1, label_one_hot=True):
        super().__init__()
        self.base_dir = base_dir
        self.is_val = is_val
        self.transform = img_transform
        self.use_pil = use_pil
        self.aug_ratio = aug_ratio
        self.label_one_hot = label_one_hot
        self.data = []
        self._load_data()

        self.aug_transform = None

        if self.aug_ratio > 1:
            self.data = self.data * self.aug_ratio
            self.aug_transform = transforms.Compose([
                transforms.RandomRotation((-10, 10)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(0, (0.1, 0.1), (0.8, 1)),
            ])

    @classmethod
    def type_to_name(cls, type_idx):
        return cls.type_idx_to_name.get(type_idx, '')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, img_label = self.data[idx]

        if self.use_pil is True:
            image = Image.open(img_path)
        else:
            image = read_image(img_path)

        if self.aug_transform is not None:
            image = self.aug_transform(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_one_hot is True:
            label = nn.functional.one_hot(
                torch.LongTensor([img_label]),
                len(self.type_idx_to_name)
            ).squeeze()
        else:
            label = img_label

        return image, label

    def _load_data(self):
        if self.is_val is True:
            image_dir = os.path.join(self.base_dir, 'val')
        else:
            image_dir = os.path.join(self.base_dir, 'train')

        for image_name in os.listdir(image_dir):
            if image_name.endswith('.jpg') or image_name.endswith('png'):
                image_path = os.path.join(image_dir, image_name)
                image_label = 0
                self.data.append((image_path, image_label))
            else:
                print(f'Warning: skip unknown image file: {image_name}')


class MstarClassificationDataset(Dataset):
    name_to_type_idx = {
        '2S1': 0,
        'BMP2': 1,
        'BRDM2': 2,
        'BTR60': 3,
        'BTR70': 4,
        'D7': 5,
        'T62': 6,
        'T72': 7,
        'ZIL131': 8,
        'ZSU23_4': 9,
    }

    type_idx_to_name = {v: k for k, v in name_to_type_idx.items()}

    def __init__(self, base_dir, is_val=False, img_transform=None, use_pil=False, aug_ratio=1, label_one_hot=True):
        super().__init__()
        self.base_dir = base_dir
        self.is_val = is_val
        self.transform = img_transform
        self.use_pil = use_pil
        self.aug_ratio = aug_ratio
        self.label_one_hot = label_one_hot
        self.data = []
        self._load_data()

        self.aug_transform = None

        if self.aug_ratio > 1:
            self.data = self.data * self.aug_ratio
            self.aug_transform = transforms.Compose([
                transforms.RandomRotation((-10, 10)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(0, (0.1, 0.1), (0.8, 1)),
            ])

    @classmethod
    def type_to_name(cls, type_idx):
        return cls.type_idx_to_name.get(type_idx, '')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, img_label = self.data[idx]

        if self.use_pil is True:
            image = Image.open(img_path)
        else:
            image = read_image(img_path)

        if self.aug_transform is not None:
            image = self.aug_transform(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_one_hot is True:
            label = nn.functional.one_hot(
                torch.LongTensor([img_label]),
                len(self.type_idx_to_name)
            ).squeeze()
        else:
            label = img_label

        return image, label

    def _load_data(self):
        if self.is_val is True:
            image_dir = os.path.join(self.base_dir, 'val')
        else:
            image_dir = os.path.join(self.base_dir, 'train')

        for type_name in os.listdir(image_dir):
            type_idx = self.name_to_type_idx.get(type_name)
            if type_idx is None:
                print(f'Warning: Ignore unknown type: {type_name}')
                continue

            for image_name in os.listdir(type_path := os.path.join(image_dir, type_name)):
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    image_path = os.path.join(type_path, image_name)
                    image_label = type_idx
                    self.data.append((image_path, image_label))
                else:
                    print(f'Warning: Skip unknown image file: {os.path.join(type_path, image_name)}')


class MstarSingleTypeDataset(MstarClassificationDataset):
    def __init__(self, base_dir, is_val=False, img_transform=None, use_pil=False, img_type=0, aug_ratio=1,
                 label_one_hot=True):
        super().__init__(base_dir, is_val=is_val, img_transform=img_transform, use_pil=use_pil, aug_ratio=aug_ratio,
                         label_one_hot=label_one_hot)
        self.img_type = img_type
        self.data = [(img_path, img_label) for (img_path, img_label) in self.data if img_label == self.img_type]

    def __getitem__(self, idx):
        img_path, img_label = self.data[idx]

        if self.use_pil is True:
            image = Image.open(img_path)
        else:
            image = read_image(img_path)

        if self.aug_transform is not None:
            image = self.aug_transform(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_one_hot is True:
            label = nn.functional.one_hot(
                torch.LongTensor([img_label]),
                len(self.type_idx_to_name)
            ).squeeze()
        else:
            label = img_label

        return image, label