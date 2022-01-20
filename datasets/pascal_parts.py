"""
Code adapted from: https://github.com/akanazawa/cmr/blob/master/data/cub.py
MIT License
Copyright (c) 2018 akanazawa
"""
import os

import cv2
import scipy.io
from tqdm import tqdm

from utils.utils import pil_loader, pad_if_smaller

cv2.setNumThreads(0)
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode

from utils import image as image_utils


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax, rmax]


def pct_area(img, bbox):
    x0, y0, x1, y1 = bbox
    image_size = img.shape
    return (x1 - x0) * (y1 - y0) / (image_size[0] * image_size[1] + 1e-7)


padding_frac = 0.05
jitter_frac = 0.05

dict_part = {'tvmonitor': ['background', 'screen'],
             'cat': ['background', 'head', 'lbleg', 'lbpa', 'lear', 'leye', 'lfleg', 'lfpa', 'neck', 'nose', 'rbleg', 'rbpa', 'rear', 'reye', 'rfleg', 'rfpa', 'tail', 'torso'],
             'person': ['background', 'hair', 'head', 'lear', 'lebrow', 'leye', 'lfoot', 'lhand', 'llarm', 'llleg', 'luarm', 'luleg', 'mouth', 'neck', 'nose', 'rear', 'rebrow', 'reye', 'rfoot', 'rhand', 'rlarm', 'rlleg', 'ruarm', 'ruleg', 'torso'],
             'motorbike': ['background', 'bwheel', 'fwheel', 'handlebar', 'headlight', 'saddle'],
             'car': ['background', 'backside', 'bliplate', 'door', 'fliplate', 'frontside', 'headlight', 'leftmirror', 'leftside', 'rightmirror', 'rightside', 'roofside', 'wheel', 'window'],
             'aeroplane': ['background', 'body', 'engine', 'lwing', 'rwing', 'stern', 'tail', 'wheel'],
             'dog': ['background', 'head', 'lbleg', 'lbpa', 'lear', 'leye', 'lfleg', 'lfpa', 'muzzle', 'neck', 'nose', 'rbleg', 'rbpa', 'rear', 'reye', 'rfleg', 'rfpa', 'tail', 'torso'],
             'bus': ['background', 'backside', 'bliplate', 'door', 'fliplate', 'frontside', 'headlight', 'leftmirror', 'leftside', 'rightmirror', 'rightside', 'roofside', 'wheel', 'window'],
             'train': ['background', 'cbackside', 'cfrontside', 'cleftside', 'coach', 'crightside', 'croofside', 'hbackside', 'head', 'headlight', 'hfrontside', 'hleftside', 'hrightside', 'hroofside'],
             'bird': ['background', 'beak', 'head', 'leye', 'lfoot', 'lleg', 'lwing', 'neck', 'reye', 'rfoot', 'rleg', 'rwing', 'tail', 'torso'],
             'horse': ['background', 'head', 'lbho', 'lblleg', 'lbuleg', 'lear', 'leye', 'lfho', 'lflleg', 'lfuleg', 'muzzle', 'neck', 'rbho', 'rblleg', 'rbuleg', 'rear', 'reye', 'rfho', 'rflleg', 'rfuleg', 'tail', 'torso'],
             'pottedplant': ['background', 'plant', 'pot'],
             'cow': ['background', 'head', 'lblleg', 'lbuleg', 'lear', 'leye', 'lflleg', 'lfuleg', 'lhorn', 'muzzle', 'neck', 'rblleg', 'rbuleg', 'rear', 'reye', 'rflleg', 'rfuleg', 'rhorn', 'tail', 'torso'],
             'bicycle': ['background', 'bwheel', 'chainwheel', 'fwheel', 'handlebar', 'headlight', 'saddle'],
             'bottle': ['background', 'body', 'cap'],
             'sheep': ['background', 'head', 'lblleg', 'lbuleg', 'lear', 'leye', 'lflleg', 'lfuleg', 'lhorn', 'muzzle', 'neck', 'rblleg', 'rbuleg', 'rear', 'reye', 'rflleg', 'rfuleg', 'rhorn', 'tail', 'torso']}


# https://github.com/micco00x/py-pascalpart
# Load annotations from .mat files creating a Python dictionary:
def load_annotations(path):
    # Get annotations from the file and relative objects:
    annotations = scipy.io.loadmat(path)["anno"]

    objects = annotations[0, 0]["objects"]

    # List containing information of each object (to add to dictionary):
    objects_list = []

    # Go through the objects and extract info:
    for obj_idx in range(objects.shape[1]):
        obj = objects[0, obj_idx]

        # Get classname and mask of the current object:
        classname = obj["class"][0]
        mask = obj["mask"]

        # List containing information of each body part (to add to dictionary):
        parts_list = []

        parts = obj["parts"]

        # Go through the part of the specific object and extract info:
        for part_idx in range(parts.shape[1]):
            part = parts[0, part_idx]
            # Get part name and mask of the current body part:
            part_name = part["part_name"][0]
            part_mask = part["mask"]

            # Add info to parts_list:
            parts_list.append({"part_name": part_name, "mask": part_mask})

        # Add info to objects_list:
        objects_list.append({"class": classname, "mask": mask, "parts": parts_list})

    return {"objects": objects_list}


class PPDataset(Dataset):
    def __init__(self, opts):
        super().__init__()

        self.opts = opts
        self.img_size = opts.input_size
        self.split = opts.split
        self.dataset_root = opts.dataset_root
        self.dataset = 'pascal-parts'

        self.jitter_frac = jitter_frac
        self.padding_frac = padding_frac
        split = opts.split
        self.masks = []
        self.images = []
        self.bbox = []

        annotation_folder = f'{opts.dataset_root}/VOCdevkit_2010/Annotations_Part/'
        images_folder = f'{opts.dataset_root}/VOCdevkit_2010/VOC2010/JPEGImages/'
        cls = opts.pascal_class
        mat_filenames = os.listdir(annotation_folder)
        voc_list = {str(s) for s in np.loadtxt(f'{opts.dataset_root}/VOCdevkit_2010/VOC2010/ImageSets/Main/{cls}_{"train" if split == "train" else "val"}.txt', dtype=str)[:, 0]}
        for idx, annotation_filename in enumerate(tqdm(mat_filenames)):
            if annotation_filename.split('.')[0] in voc_list:
                annotations = load_annotations(os.path.join(annotation_folder, annotation_filename))
                for obj in annotations["objects"]:
                    if obj["class"] == cls:
                        bbox = bbox2(obj['mask'])
                        mask = np.zeros_like(obj['mask'])
                        if pct_area(obj['mask'], bbox) > (0.20 if split == 'test' else 0.10):
                            for body_part in obj["parts"][::-1]:
                                instance_mask = body_part["mask"].astype(np.uint8)
                                part_name = body_part["part_name"].split('_')[0]
                                mask = mask * (1 - instance_mask) + instance_mask * dict_part[obj["class"]].index(part_name)

                            self.bbox.append(bbox)
                            self.images.append(images_folder + annotation_filename[:annotation_filename.rfind(".")] + ".jpg")
                            self.masks.append(mask.astype(np.uint8))

        print(f"Total {split}: {len(voc_list)} {cls} {split}: {len(self.images)}")

    @staticmethod
    def only_file_names(lst):
        return [e['file_name'] for e in lst]

    def forward_img(self, index):
        path = self.images[index]
        img = pil_loader(path, 'RGB')
        mask = self.masks[index]
        mask = Image.fromarray(mask)
        img = np.array(img)
        mask = np.array(mask)
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(mask, 2)
        h, w, _ = mask.shape

        bbox = self.bbox[index]
        if self.split == 'train':
            bbox = image_utils.peturb_bbox(bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(bbox, pf=self.padding_frac, jf=0)

        bbox = image_utils.square_bbox(bbox)

        img, mask = self.crop_image(img, mask, bbox)
        # scale image, and mask. And scale kps.
        img, mask = self.scale_image(img, mask)

        # Mirror image on random.
        if self.split == 'train':
            img, mask = self.mirror_image(img, mask)

        img = Image.fromarray(img.astype(np.uint8))
        mask = np.asarray(mask, np.uint8)
        return img, mask, path

    def crop_image(self, img, mask, bbox):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        return img, mask

    def scale_image(self, img, mask):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)

        mask_scale, _ = image_utils.resize_img(mask, scale, interpolation=cv2.INTER_NEAREST)
        mask_scale = np.expand_dims(mask_scale, 2)

        img_scale = pad_if_smaller(img_scale, self.img_size)
        mask_scale = pad_if_smaller(mask_scale, self.img_size)
        return img_scale, mask_scale

    def mirror_image(self, img, mask):
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            return img_flip, mask_flip
        else:
            return img, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, seg, img_path = self.forward_img(index)
        mask = (seg != 0).astype(np.uint8)
        elem = {
            'img': img,
            'mask': mask,
            'seg': seg,
            'inds': index,
            'img_path': img_path,
        }
        return elem
