"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import cv2
import wandb

from utils.utils import batch_get_centers

softmax = nn.Softmax(dim=1)

class BatchColorize(object):
    def __init__(self, n=40):
        self.cmap = color_map(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((size[0], 3, size[1], size[2]), dtype=np.float32)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[:,0][mask] = self.cmap[label][0]
            color_image[:,1][mask] = self.cmap[label][1]
            color_image[:,2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[:,0][mask] = color_image[:,1][mask] = color_image[:,2][mask] = 255

        return color_image

def color_map(N=256, normalized=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_centers(pred_softmax):
    input_size = pred_softmax.shape[-1]
    pos_x = torch.arange(input_size).view(1, 1, 1, -1).repeat(1, 1, input_size, 1).to(pred_softmax.device)
    pos_y = torch.arange(input_size).view(1, 1, -1, 1).repeat(1, 1, 1, input_size).to(pred_softmax.device)
    fac = 1.0 / torch.clamp_min(pred_softmax.sum(3).sum(2), 1.0)
    center_of_mass_x = (pred_softmax * pos_x).sum(3).sum(2) * fac
    center_of_mass_y = (pred_softmax * pos_y).sum(3).sum(2) * fac
    pred_parts_raw = torch.cat([center_of_mass_x.unsqueeze(2), center_of_mass_y.unsqueeze(2)], dim=2)
    pred_parts = 2 * pred_parts_raw / pred_softmax.size(2) - 1  # normalize by image_size
    return pred_parts

def Batch_Draw_GT_Landmarks(imgs, pred, lms):

    B,_,H,W = imgs.shape
    C = lms.shape[1]
    cmap = color_map(40,normalized=False)
    imgs_cv2 = imgs.detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8)

    for b in range(B):
        for c in range(C):
            x_c = int(lms[b][c][0])
            y_c = int(lms[b][c][1])

            img = imgs_cv2[b].copy()
            cv2.drawMarker(img, (x_c,y_c), (int(cmap[c+1][0]), int(cmap[c+1][1]), int(cmap[c+1][2])), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
            imgs_cv2[b] = img

    return imgs_cv2.transpose(0,3,1,2)

def Batch_Draw_Bboxes(imgs, bboxes):

    B,C,H,W = imgs.shape
    imgs_cv2 = imgs.detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8)
    for b in range(B):
        x,y,w,h = bboxes[b]
        img = imgs_cv2[b].copy()
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        imgs_cv2[b] = img

    return imgs_cv2.transpose(0,3,1,2)


def Batch_Draw_Landmarks(imgs, pred, sm=True):

    B,C,H,W = pred.shape
    cmap = color_map(40, normalized=False)

    if sm:
        pred_softmax = torch.softmax(pred, dim=1)
    else:
        pred_softmax = pred

    imgs_cv2 = imgs.detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8)

    for b in range(B):
        centers = get_centers(pred_softmax)
        for c in range(1,C):
            x_c = centers[b, c, 0]
            y_c = centers[b, c, 1]
            x_c = (x_c+1.0)/2*W
            y_c = (y_c+1.0)/2*H
            img = imgs_cv2[b].copy()
            cv2.drawMarker(img, (x_c.int().item(),y_c.int().item()), (int(cmap[c][0]), int(cmap[c][1]), int(cmap[c][2])), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
            imgs_cv2[b] = img

    return imgs_cv2.transpose(0,3,1,2)


class Visualizer(object):
    def __init__(self, args, viz=None):

        self.exp_name = wandb.run.name
        self.vis_interval = args.vis_interval
        self.colorize = BatchColorize(args.num_classes)

        self.args = args


    def vis_image(self, setting, i_iter, name, tps_imgs, mean):
        if i_iter % self.vis_interval == 0 :
            i_shape = tps_imgs.shape
            mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
            tps_imgs_viz = torch.clamp(tps_imgs+mean_tensor, 0.0, 255.0)
            tps_imgs_viz = vutils.make_grid(tps_imgs_viz/255.0, normalize=False, scale_each=False)
            wandb.log({f'{setting}/{name}': wandb.Image(tps_imgs_viz)}, step=i_iter)

    def vis_image_pred(self, setting, i_iter, name, tps_imgs, tps_pred, mean):
        if i_iter % self.vis_interval == 0 :
            i_shape = tps_imgs.shape
            mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
            tps_pred = tps_pred.detach().cpu().float().numpy()
            tps_pred = np.asarray(np.argmax(tps_pred, axis=1), dtype=np.int)
            tps_pred = self.colorize(tps_pred)
            tps_pred = vutils.make_grid(torch.tensor(tps_pred), normalize=False, scale_each=False)
            tps_imgs_viz = torch.clamp(tps_imgs+mean_tensor, 0.0, 255.0)
            tps_imgs_viz = vutils.make_grid(tps_imgs_viz/255.0, normalize=False, scale_each=False)
            tps_imgs_viz = (tps_imgs_viz + tps_pred)/2
            wandb.log({f'{setting}/{name}': wandb.Image(tps_imgs_viz)}, step=i_iter)


    def vis_images(self, setting, i_iter, imgs, tps_items, saliency_imgs, edge_imgs, mean, pred):
        if i_iter % self.vis_interval == 0 :
            log_dict = {}
            i_shape = imgs.shape
            mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
            imgs_viz = torch.clamp(imgs+mean_tensor, 0.0, 255.0)
            self.imgs_viz = imgs_viz
            imgs_viz_grid = vutils.make_grid(imgs_viz/255.0, normalize=False, scale_each=False)
            self.imgs_viz_grid = imgs_viz_grid
            log_dict[f'{setting}/Input'] = wandb.Image(imgs_viz_grid)

            tps_imgs, tps_pred = tps_items
            tps_pred = tps_pred.detach().cpu().float().numpy()
            tps_pred = np.asarray(np.argmax(tps_pred, axis=1), dtype=np.int)
            tps_pred = self.colorize(tps_pred)
            tps_pred = vutils.make_grid(torch.tensor(tps_pred), normalize=False, scale_each=False)
            tps_imgs_viz = torch.clamp(tps_imgs+mean_tensor, 0.0, 255.0)
            tps_imgs_viz = vutils.make_grid(tps_imgs_viz/255.0, normalize=False, scale_each=False)
            tps_imgs_viz = (tps_imgs_viz + tps_pred)/2
            log_dict[f'{setting}/Transformed'] = wandb.Image(tps_imgs_viz)

            # saliency
            if saliency_imgs is not None:
                sal_viz = torch.clamp(saliency_imgs.float().unsqueeze(dim=1)*255.0, 0.0, 255.0)
                sal_viz = vutils.make_grid(sal_viz/255.0, normalize=False, scale_each=False)
                self.sal_viz = sal_viz
                log_dict[f'{setting}/Saliency'] = wandb.Image(sal_viz)

            # edges
            if edge_imgs is not None:
                edge_viz = torch.clamp(edge_imgs.float().unsqueeze(dim=1)*255.0, 0.0, 255.0)
                edge_viz = vutils.make_grid(edge_viz/255.0, normalize=False, scale_each=False)
                log_dict[f'{setting}/Edge'] = wandb.Image(edge_viz)

            # landmarks
            lm_viz = Batch_Draw_Landmarks(imgs_viz, pred[:,1:], sm=False)
            lm_viz = torch.tensor(lm_viz.astype(np.float32))
            lm_viz = vutils.make_grid(lm_viz/255.0, normalize=False, scale_each=False)
            log_dict[f'{setting}/Landmark'] = wandb.Image(lm_viz)


            pred = pred.detach().cpu().float().numpy()
            pred = np.asarray(np.argmax(pred, axis=1), dtype=np.int)
            pred = self.colorize(pred)
            pred = vutils.make_grid(torch.tensor(pred), normalize=False, scale_each=False)
            pred_viz = (self.imgs_viz_grid + pred)/2
            log_dict[f'{setting}/Part Map'] = wandb.Image(pred_viz)

            # saliency
            if saliency_imgs is not None:
                pred = (sal_viz + pred)/2
                log_dict[f'{setting}/Part Map sal'] = wandb.Image(pred)

            wandb.log(log_dict, step=i_iter)


    def vis_part_heatmaps(self, setting, i_iter, response_maps, threshold=0.5, prefix=''):
        if i_iter % self.vis_interval == 0:
            log_dict = {}
            B,K,H,W = response_maps.shape
            part_response = np.zeros((B,K,H,W,3)).astype(np.uint8)

            for b in range(B):
                for k in range(K):
                    response_map = response_maps[b,k,...].cpu().numpy()
                    response_map = cv2.applyColorMap((response_map*255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,::-1] # BGR->RGB
                    part_response[b,k,:,:,:] = response_map.astype(np.uint8)

            part_response = part_response.transpose(0,1,4,2,3)
            part_response = torch.tensor(part_response.astype(np.float32))
            for k in range(K):
                map_viz_single = vutils.make_grid(part_response[:,k,:,:,:].squeeze()/255.0, normalize=False, scale_each=False)
                log_dict[f'{setting}/{prefix} PART {k}'] = wandb.Image(map_viz_single)

            # color segmentation
            response_maps_np = response_maps.cpu().numpy()
            response_maps_np = np.concatenate((np.ones((B,1,H,W))*threshold, response_maps_np), axis=1)
            response_maps_np = np.asarray(np.argmax(response_maps_np, axis=1), dtype=np.int)
            response_maps_np = self.colorize(response_maps_np)
            response_maps_np = vutils.make_grid(torch.tensor(response_maps_np), normalize=False, scale_each=False)
            response_maps_np_viz = (self.imgs_viz_grid + response_maps_np)/2
            log_dict[f'{setting}/{prefix} Map'] = wandb.Image(response_maps_np_viz)

            if self.sal_viz is not None:
                pred_part = (self.sal_viz + response_maps_np) / 2
                log_dict[f'{setting}/{prefix} Map Sal'] = wandb.Image(pred_part)

            wandb.log(log_dict, step=i_iter)

    def vis_landmarks(self, setting, i_iter, imgs, mean, pred, lms):
        if i_iter % self.vis_interval == 0 :

            i_shape = imgs.shape
            mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
            imgs_viz = torch.clamp(imgs+mean_tensor, 0.0, 255.0)
            self.imgs_viz = imgs_viz

            lm_viz = Batch_Draw_GT_Landmarks(imgs_viz, pred, lms)
            lm_viz = torch.tensor(lm_viz.astype(np.float32))

            lm_viz = vutils.make_grid(lm_viz/255.0, normalize=False, scale_each=False)
            wandb.log({f'{setting}/Landmark_GT': wandb.Image(lm_viz)}, step = i_iter)

    def vis_bboxes(self, setting, i_iter, bboxes):
        if i_iter % self.vis_interval == 0 :

            bbox_viz = Batch_Draw_Bboxes(self.imgs_viz, bboxes)
            bbox_viz = torch.tensor(bbox_viz.astype(np.float32))

            bbox_viz = vutils.make_grid(bbox_viz/255.0, normalize=False, scale_each=False)

            wandb.log({f'{setting}/BBOX_GT': wandb.Image(bbox_viz)}, step = i_iter)

    def vis_losses(self, i_iter, losses, names):
        wandb.log({('data/' + names[i]): loss for i, loss in enumerate(losses)}, step=i_iter)

    def vis_embeddings(self, i_iter, part_feat_list_all):
        # check visualization interval
        if i_iter % (self.vis_interval*10) != 0:
            return

        feat_list = []
        img_list = []
        label_list = []

        for i in range(len(part_feat_list_all)):
            # i: img index
            for j in range(len(part_feat_list_all[i])):
                # j : part index
                if part_feat_list_all[i][j].shape[0] != 0 :
                    label_list.append(j)
                    img_list.append(self.imgs_viz[i:i+1,...])
                    feat_list.append(part_feat_list_all[i][j].detach().cpu())

        label_tensor = torch.tensor(label_list)
        img_tensor = torch.cat(img_list, dim=0)
        feat_tensor = torch.cat(feat_list, dim=0)
        print('show embedding iter {}'.format(i_iter))
        self.tb_writer.add_embedding(feat_tensor,
                                     tag='part_feature',
                                     metadata=label_tensor,
                                     label_img=img_tensor,
                                     global_step=i_iter)
