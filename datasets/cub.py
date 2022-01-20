"""
Code adapted from: https://github.com/akanazawa/cmr/blob/master/data/cub.py
MIT License

Copyright (c) 2018 akanazawa
"""

import os.path as osp

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import image as image_utils
from torchvision.transforms import functional as TF, InterpolationMode
from utils import transformations

padding_frac = 0.05
jitter_frac = 0.05

class CUBDataset(Dataset):
    def __init__(self, opts):
        super().__init__()

        self.opts = opts
        self.img_size = opts.input_size
        self.jitter_frac = jitter_frac
        self.padding_frac = padding_frac
        self.split = opts.split
        self.unsup_mask = opts.unsup_mask

        self.data_dir = f'{opts.dataset_root}/CUB/CUB_200_2011/'
        self.data_cache_dir = f'{self.data_dir}/cachedir/cub'  # https://github.com/akanazawa/cmr/issues/3#issuecomment-451757610

        self.img_dir = osp.join(self.data_dir, 'images')
        self.pmask_dir = self.data_dir.replace('CUB_200_2011', 'pseudolabels')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % self.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % self.split)
        # import ipdb; ipdb.set_trace()
        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import pdb; pdb.set_trace()

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1;

        self.labels = [int(self.anno[index].rel_path.split('.')[0]) for index in range(len(self.anno))]
        if opts.single_class is not None:
            idx = [i for i, c in enumerate(self.labels) if c == opts.single_class]
            self.anno = [self.anno[i] for i in idx]
            self.anno_sfm = [self.anno_sfm[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)

    def forward_img(self, index):
        data = self.anno[index]
        data_sfm = self.anno_sfm[index]

        # sfm_pose = (sfm_c, sfm_t, sfm_r)
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0,1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = osp.join(self.img_dir, str(data.rel_path))
        #img_path = img_path.replace("JPEG", "jpg")
        img = np.array(Image.open(img_path))

        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        if self.unsup_mask:
            if self.split == 'train':
                mask = data.mask
            else:
                mask = TF.resize(torch.from_numpy(np.array(Image.open(osp.join(self.pmask_dir, str(data.rel_path).replace('.jpg', '.png'))))).unsqueeze(0), img.shape[:2], interpolation=InterpolationMode.NEAREST).squeeze(0).numpy()/255.
        else:
            mask = data.mask
        mask = np.expand_dims(mask, 2)
        h,w,_ = mask.shape

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        # Peturb bbox
        if self.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)

        # scale image, and mask. And scale kps.
        img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)

        # Mirror image on random.
        if self.split == 'train':
           img, mask, kp, sfm_pose = self.mirror_image(img, mask, kp, sfm_pose)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        img = Image.fromarray(np.asarray(img, np.uint8))
        mask = np.asarray(mask, np.float32)
        return img, kp_norm, mask, sfm_pose, img_path

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                           2 * (kp[:, 1] / img_h) - 1,
                           kp[:, 2]]).T
        sfm_pose[0] *= (1.0/img_w + 1.0/img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp

        return new_kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]
        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]
        return img, mask, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        # mask_scale, _ = image_utils.resize_img(mask, scale)
        mask_scale, _ = image_utils.resize_img(mask, scale, interpolation=cv2.INTER_NEAREST)
        kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose):
        kp_perm = self.kp_perm
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            # Flip kps.
            new_x = img.shape[1] - kp[:, 0] - 1
            kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
            kp_flip = kp_flip[kp_perm, :]
            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            return img_flip, mask_flip, kp_flip, sfm_pose
        else:
            return img, mask, kp, sfm_pose

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        img, kp, mask, sfm_pose, img_path = self.forward_img(index)
        sfm_pose[0].shape = 1
        mask = np.expand_dims(mask, 2)

        elem = {
            'img': img,
            'kp': kp,
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),
            'inds': index,
            'label': self.labels[index],
            'img_path': img_path,
        }

        return elem
