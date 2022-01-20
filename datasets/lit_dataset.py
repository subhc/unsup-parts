import random

import torch
from PIL import ImageFilter, ImageDraw
from skimage import color
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torchvision.transforms import functional as TF, InterpolationMode

IMG_MEAN = torch.from_numpy(np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)).view(3, 1, 1)


class ToTensorDeepLabNormalized:
    def __call__(self, img):
        img = torch.from_numpy(np.array(img, np.int32, copy=False))
        img = img.permute((2, 0, 1)).contiguous()
        return img - IMG_MEAN


def to_tensor(img):
    img = torch.from_numpy(np.array(img, np.int32, copy=False))
    img = img.permute((2, 0, 1)).contiguous()
    return img

class ToLAB:
    def __call__(self, img):
        img = color.rgb2lab(img)
        img = torch.from_numpy(np.array(img, np.float32, copy=False))
        img = img.permute((2, 0, 1)).contiguous()
        return img


class PhotometricAug(object):
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        n = random.randint(0, 2)
        if n == -1:
            transformed_image = TF.invert(img.copy())
        elif n == 0:
            transformed_image = img.copy().convert('L').filter(ImageFilter.FIND_EDGES).convert('RGB')
        elif n == -2:
            transformed_image = img.copy()
            draw = ImageDraw.Draw(transformed_image)
            width, height = img.size
            x0 = random.randint(0, width-1)
            y0 = random.randint(0, height-1)
            wl = (width-x0)//4
            hl = (height-y0)//4
            if wl > 5 and hl > 5:
                x1 = min(width, x0 + random.randint(1, wl))
                y1 = min(height, y0 + random.randint(1, hl))
                draw.rectangle(((x0, y0), (x1, y1)), fill="black")

        else:
            transformed_image = self.transform(img)

        return transformed_image

class LitDataset(Dataset):
    def __init__(self, dataset, use_lab=True):
        self.dataset = dataset
        self.use_lab = use_lab
        self.dlcj_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            ToTensorDeepLabNormalized()
        ])

        self.dl_transform = transforms.Compose([
            ToTensorDeepLabNormalized()
        ])

        self.lab = transforms.Compose([
            ToLAB()
        ])

        self.vgg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        example = self.dataset[idx]
        img = example['img'].copy()
        elem = {
            'img_vgg': self.vgg_transform(img),
            'img_rec': self.lab(img) if self.use_lab else to_tensor(img)-128,
            'img': self.dl_transform(img),
            'img_cj1': self.dlcj_transform(img),
            'img_cj2': self.dlcj_transform(img),
            'mask': torch.from_numpy(example['mask']).permute(2, 0, 1),
            'seg': torch.from_numpy(example['seg']).permute(2, 0, 1) if 'seg' in example else [],
            'inds': example['inds'] if 'inds' in example else [],
            'kp': example['kp'] if 'kp' in example else [],
            'label': example['label'] if 'label' in example else [],
            'img_path': example['img_path'] if 'img_path' in example else [],
            'landmarks': example['landmarks'] if 'landmarks' in example else [],
        }
        return elem

    def __len__(self):
        return len(self.dataset)
