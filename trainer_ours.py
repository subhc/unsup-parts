"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Modified by @subhc
"""

import os.path as osp
import random

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import loss
from datasets.lit_dataset import LitDataset
from models.feature_extraction import FeatureExtraction
from models.model_factory import model_generator
from tps.rand_tps import RandTPS
from utils.utils import adjusted_rand_score_overflow
from visualize import Visualizer

IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, 100000, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


class Trainer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print("Nondet")

        # create network
        model, optimizer_state_dict = model_generator(args, add_bg_mask=False)
        model.train()
        model.cuda(args.gpu)
        self.model = model

        # Initialize spatial/color transform for Equuivariance loss.
        self.tps = RandTPS(args.input_size, args.input_size,
                           batch_size=args.batch_size,
                           sigma=args.tps_sigma,
                           border_padding=args.eqv_border_padding,
                           random_mirror=args.eqv_random_mirror,
                           random_scale=(args.random_scale_low,
                                         args.random_scale_high),
                           mode=args.tps_mode).cuda(args.gpu)

        # KL divergence loss for equivariance
        self.kl = nn.KLDivLoss(reduction='none').cuda(args.gpu)

        # loss/ bilinear upsampling
        self.interp = nn.Upsample(
            size=(args.input_size, args.input_size), mode='bilinear', align_corners=True)

        # Initialize feature extractor and part basis for the semantic consistency loss.
        if int(args.ref_layer1[-3]+args.ref_layer1[-1]) <= int(args.ref_layer2[-3]+args.ref_layer2[-1]):
            last_layer = args.ref_layer1+','+args.ref_layer2
            weights = [args.ref_weight1, args.ref_weight2]
        else:
            last_layer = args.ref_layer2+','+args.ref_layer1
            weights = [args.ref_weight2, args.ref_weight1]

        self.zoo_feat_net = FeatureExtraction(
            feature_extraction_cnn=args.ref_net, normalization=args.ref_norm, last_layer=last_layer, weights=weights, gpu=args.gpu)
        self.zoo_feat_net.eval()

        # Initialize optimizers.
        self.optimizer_seg = optim.SGD(self.model.optim_parameters(args),
                                       lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        if optimizer_state_dict is not None:
            # resume training from checkpoint
            self.optimizer_seg.load_state_dict(optimizer_state_dict)
        else:
            self.optimizer_seg.zero_grad()

        # visualizor
        self.viz = Visualizer(args)

        self.register_buffer('pos_x', torch.arange(self.args.input_size).view(1, 1, 1, -1).repeat(1, 1, self.args.input_size, 1))
        self.pos_x = self.pos_x.cuda(args.gpu)
        self.register_buffer('pos_y', torch.arange(self.args.input_size).view(1, 1, -1, 1).repeat(1, 1, 1, self.args.input_size).cuda(args.gpu))
        self.pos_y = self.pos_y.cuda(args.gpu)

    def step(self, batch, current_step):
        loss_eqv_value = 0
        loss_sc_value = 0
        loss_rgb_value = 0
        loss_contrastive_value = 0

        self.model.train()

        self.optimizer_seg.zero_grad()
        adjust_learning_rate(self.optimizer_seg, current_step, self.args)

        images = batch['img_cj1'].cuda(self.args.gpu)
        mask = batch['mask'].cuda(self.args.gpu).float() if 'mask' in batch.keys() else None
        images_vgg = batch['img_vgg'].cuda(self.args.gpu)
        images_rec = batch['img_rec'].cuda(self.args.gpu)

        _, _, pred_low, _ = self.model(images)
        pred = self.interp(pred_low)

        zoo_feat = self.get_zoo_feat(images_vgg)
        zoo_feat = zoo_feat * mask

        pred_softmax = torch.softmax(pred, dim=1)

        basis = torch.einsum('brhw, bchw -> brc', pred_softmax * mask, zoo_feat)
        basis /= einops.reduce(pred_softmax * mask, 'b r h w -> b r 1', 'sum') + 1e-7
        loss_sc = 0
        for pred_sm_, zoo_feat_, mask_ in zip(pred_softmax * mask, zoo_feat, mask):
            loss_sc += loss.consistency_loss(pred_sm_.unsqueeze(0), zoo_feat_.unsqueeze(0), mask_.unsqueeze(0))
        loss_sc_value += self.args.lambda_sc * loss_sc.data.cpu().numpy()

        loss_rgb = loss.consistency_loss(pred_softmax * mask, images_rec, mask)
        loss_rgb_value += self.args.lambda_rgb * loss_rgb.data.cpu().numpy()

        # contrastive_loss
        loss_contrastive = loss.contrastive_loss(basis[:, :, -self.args.layer_len:] if self.args.layer_len > 0 else basis, self.args.temperature)
        loss_contrastive_value += self.args.lambda_contrastive * loss_contrastive.data.cpu().numpy()

        # Equivariance Loss
        images_cj = batch['img_cj2'].cuda(self.args.gpu)

        self.tps.reset_control_points()

        images_tps = self.tps(images_cj)

        mask_tps = self.tps(mask.float(), padding_mode='zeros')

        _, _, pred_low_tps, _ = self.model(images_tps)
        pred_tps = self.interp(pred_low_tps)
        pred_d = pred.detach()
        pred_d.requires_grad = False

        # pred_d_rrc = TF.resized_crop(pred_d, *rrc_params, self.args.input_size)
        pred_tps_org = self.tps(pred_d, padding_mode='zeros')

        loss_eqv = self.kl(F.log_softmax(pred_tps, dim=1),
                           F.softmax(pred_tps_org, dim=1))
        loss_eqv = (loss_eqv * mask_tps).flatten(1).sum(1) / (mask_tps.flatten(1).sum(1) + 1e-7)
        loss_eqv = loss_eqv.mean()
        loss_eqv_value += self.args.lambda_eqv * loss_eqv.data.cpu().numpy()

        # sum all loss terms
        total_loss = self.args.lambda_eqv * loss_eqv \
                     + self.args.lambda_sc * loss_sc \
                     + self.args.lambda_rgb * loss_rgb \
                     + self.args.lambda_contrastive * loss_contrastive

        total_loss.backward()

        # clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradients)
        self.optimizer_seg.step()
        if current_step % 100 == 0:
            # visualize loss curves
            self.viz.vis_losses(current_step,
                                [loss_eqv_value,
                                 loss_sc_value, loss_rgb_value, loss_contrastive_value],
                                ['loss_eqv', 'loss_sc', 'loss_rgb', 'loss_contrastive'])
            print('exp = {}'.format(osp.join(self.args.snapshot_dir, wandb.run.name)))
            print(('iter = {:8d}/{:8d}, ' +
                   'loss_eqv = {:.3f}, ' +
                   'loss_sc = {:.3f}, ' +
                   'loss_rgb = {:.3f}, ' +
                   'loss_contrastive = {:.3f}')
                  .format(current_step, self.args.num_steps,
                          loss_eqv_value,
                          loss_sc_value,
                          loss_rgb_value,
                          loss_contrastive_value))

    def visualize(self, setting, current_step, batch):

        self.model.eval()

        with torch.no_grad():
            images_cpu = batch['img']
            mask = batch['mask'].cuda(self.args.gpu) if 'mask' in batch.keys() else None
            labels = None if mask is None else batch['mask'][:, 0]
            edges = batch['edge'] if 'edge' in batch.keys() else None

            if 'landmarks' in batch.keys():
                landmarks = batch['landmarks']
            elif 'kp' in batch.keys():
                landmarks = self.args.input_size(batch['kp']+1)/2
            else:
                landmarks = None

            bbox = batch['bbox'] if 'bbox' in batch.keys() else None

            images = images_cpu.cuda(self.args.gpu)
            _, _, pred_low, pred_mask_low = self.model(images)
            pred = self.interp(pred_low)

            images_cj = batch['img_cj2'].cuda(self.args.gpu)

            self.tps.reset_control_points()
            images_tps = self.tps(images_cj)
            mask_tps = self.tps(mask.float(), padding_mode='zeros')
            pred_low_tps = self.model(images_tps)[2]
            pred_tps = self.interp(pred_low_tps) * mask_tps

            output_softmax = torch.softmax(pred, dim=1)
            pred_softmax = torch.cat([1 - mask, output_softmax * mask], dim=1)
            part_softmax = pred_softmax[:, 1:, :, :]
            # normalize
            part_softmax /= part_softmax.max(dim=3, keepdim=True)[
                0].max(dim=2, keepdim=True)[0]
            self.viz.vis_images(setting, current_step, images_cpu, (images_tps.cpu(), pred_tps),
                                labels, edges, IMG_MEAN, pred=pred_softmax.float())
            self.viz.vis_part_heatmaps(
                setting, current_step, part_softmax, threshold=0.1, prefix='pred')
            self.viz.vis_image(setting, current_step, 'Transformed CJ1', batch['img_cj1'], IMG_MEAN)
            self.viz.vis_image(setting, current_step, 'Transformed CJ2', batch['img_cj2'], IMG_MEAN)
            # self.viz.vis_image(setting, current_step, 'pred_mask', pred_mask.cpu(), np.array([0, 0, 0]))
            if len(batch['seg']) > 0:
                self.viz.vis_image(setting, current_step, 'pred_mask', batch['seg'].cpu(), np.array([0, 0, 0]))
            if len(landmarks) > 0:
                self.viz.vis_landmarks(setting, current_step, images_cpu, IMG_MEAN, pred_softmax, landmarks)
            if bbox is not None:
                self.viz.vis_bboxes(current_step, bbox)

    def get_zoo_feat(self, images_zoo):
        with torch.no_grad():
            zoo_feats = self.zoo_feat_net(images_zoo)
            zoo_feat = torch.cat([self.interp(zoo_feat) for zoo_feat in zoo_feats], dim=1)
        return zoo_feat

    def save_model(self, path, iter):
        torch.save({
            'iter': iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer_seg.state_dict(),
            }, path)

    def log_ari(self, testloader, i_iter):
        self.model.eval()
        gts = []
        preds = []
        for batch in tqdm(testloader):
            images_cpu = batch['img']
            gt = batch['seg']
            images = images_cpu.cuda(self.args.gpu)
            _, _, pred_low, _ = self.model(images)
            pred = self.interp(pred_low)

            pred_argmax = pred.argmax(dim=1)
            gts.append(gt.type(torch.int8))
            preds.append(pred_argmax.cpu().type(torch.int8))

        gts = torch.cat(gts, 0).flatten()
        preds = torch.cat(preds, 0).flatten()

        preds = preds[gts != 0]
        gts = gts[gts != 0]

        ari = adjusted_rand_score_overflow(preds, gts)
        nmi = normalized_mutual_info_score(preds, gts)
        print(f"ARI: {ari * 100: .2f}, NMI: {nmi * 100: .2f}")
        wandb.log({'data/nmi': nmi * 100}, step=i_iter)
        wandb.log({'data/ari': ari * 100}, step=i_iter)

    def log_nmi(self, trainloader, testloader, i_iter):
        self.model.eval()
        data = {}
        for k, v in self.get_pred_and_gt(trainloader).items():
            data[f"train_{k}"] = v
        for k, v in self.get_pred_and_gt(testloader).items():
            data[f"val_{k}"] = v

        nmi = normalized_mutual_info_score(data[f"val_nmi_gt"], data[f"val_nmi_pred"])
        ari = adjusted_rand_score(data[f"val_nmi_gt"], data[f"val_nmi_pred"])
        errors = np.zeros(3)
        for i in range(3):
            errors[i] = self.kp_evaluation(data, class_id=i+1)
        print(f"NMI: {nmi * 100: .2f} ARI: {ari * 100: .2f} LR1: {errors[0] * 100: .2f} LR2: {errors[1] * 100: .2f} LR3: {errors[2] * 100: .2f}")
        wandb.log({'data/nmi': nmi * 100, 'data/ari': ari * 100, 'data/regress_cls_1': errors[0] * 100, 'data/regress_cls_2': errors[1] * 100, 'data/regress_cls_3': errors[2] * 100}, step=i_iter)

    def get_pred_and_gt(self, dataset):
        loader = DataLoader(LitDataset(dataset), batch_size=8, shuffle=False, num_workers=5, drop_last=False)
        with torch.no_grad():
            all_preds = []
            all_gts = []
            all_visible = []
            all_labels = []
            all_nmi_preds = []
            all_nmi_gts = []
            for batch in tqdm(loader):
                # deal with parts and cropping
                image = batch['img'].cuda(self.args.gpu)
                mask = batch['mask'].cuda(self.args.gpu)
                parts = batch["kp"].float()
                res = self.model(image.cuda(self.args.gpu))
                part_name_mat = torch.softmax(self.interp(res[2]), dim=1)*mask
                fac = 1.0 / torch.clamp_min(part_name_mat.sum(3).sum(2), 1.0)
                center_of_mass_x = (part_name_mat * self.pos_x).sum(3).sum(2) * fac
                center_of_mass_y = (part_name_mat * self.pos_y).sum(3).sum(2) * fac
                pred_parts_raw = torch.cat([center_of_mass_x.unsqueeze(2), center_of_mass_y.unsqueeze(2)], dim=2)
                pred_parts = pred_parts_raw / part_name_mat.size(2)  # normalize by image_size
                gt_parts = (parts[:, :, :2] + 1) / 2.
                all_preds.append(pred_parts.cpu())
                all_gts.append(gt_parts.cpu())
                all_visible.append(parts[:, :, 2].cpu())
                all_labels.append(batch["label"].cpu())

                visible = parts[:, :, 2] > 0.5
                points = parts[:, :, :2].unsqueeze(2)
                part_name_mat = self.interp(res[2])
                pred_parts_loc = F.grid_sample(part_name_mat.float().cpu(), points, mode='nearest', align_corners=False)
                pred_parts_loc = torch.argmax(pred_parts_loc, dim=1).squeeze(2)
                pred_parts_loc = pred_parts_loc[visible]
                all_nmi_preds.append(pred_parts_loc.cpu().numpy())
                gt_parts_loc = torch.arange(parts.shape[1]).unsqueeze(0).repeat(parts.shape[0], 1)
                gt_parts_loc = gt_parts_loc[visible]
                all_nmi_gts.append(gt_parts_loc.cpu().numpy())

            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_gts = torch.cat(all_gts, dim=0).numpy()
            all_visible = torch.cat(all_visible, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            all_nmi_preds = np.concatenate(all_nmi_preds, axis=0)
            all_nmi_gts = np.concatenate(all_nmi_gts, axis=0)

        return {"pred": all_preds, "gt": all_gts,
                "nmi_pred": all_nmi_preds, "nmi_gt": all_nmi_gts,
                "visible": all_visible, "labels": all_labels, }

    def kp_evaluation(self, data, class_id=None):
        # https://github.com/NVlabs/SCOPS/blob/master/evaluation/face_evaluation_wild.py
        test_fit_kp = np.zeros_like(data["val_gt"])

        train_pred_flat = data["train_pred"].reshape(data["train_pred"].shape[0], -1)
        val_pred_flat = data["val_pred"].reshape(data["val_pred"].shape[0], -1)

        for i in range(data["train_gt"].shape[1]):
            scaler_pred = StandardScaler()
            scaler_gt = StandardScaler()

            train_vis = data["train_visible"][:, i] > 0.5
            scaler_pred.fit(train_pred_flat[train_vis])
            scaler_gt.fit(data["train_gt"][train_vis, i, :])

            train_pred_kp_flat_transform = scaler_pred.transform(train_pred_flat[train_vis])
            train_gt_kp_flat_transform = scaler_gt.transform(data["train_gt"][train_vis, i, :])

            mdl = LinearRegression(fit_intercept=False)
            mdl.fit(train_pred_kp_flat_transform, train_gt_kp_flat_transform)

            # test
            test_vis = data["val_visible"][:, i] > 0.5
            test_pred_kp_flat_transform = scaler_pred.transform(val_pred_flat[test_vis])
            test_fit_kp[test_vis, i, :] = scaler_gt.inverse_transform(mdl.predict(test_pred_kp_flat_transform))
        mean_error_test = self.mean_error(test_fit_kp, data, "val", class_id)
        return mean_error_test

    @staticmethod
    def mean_error(fit_kp, data, mode, class_id=None):
        gt = data[f"{mode}_gt"]
        visible = data[f"{mode}_visible"]
        diff = (fit_kp - gt)
        err = np.linalg.norm(diff, axis=2)
        err *= visible
        if class_id is not None:
            class_mask = data[f"{mode}_labels"] == class_id
            err = err[class_mask]
            visible = visible[class_mask]
        return err.sum() / visible.sum()
