import os
from argparse import ArgumentParser

import matplotlib
import pandas as pd
from sklearn import linear_model
from tqdm import trange

matplotlib.use('Agg')
import yaml
import numpy as np

import torch
from torch.utils import data
import torch.nn.functional as F

def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1)) / x_max * 2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T / y_max * 2 - 1.0

    x_map = torch.from_numpy(x_map.astype(np.float32))
    y_map = torch.from_numpy(y_map.astype(np.float32))

    return x_map, y_map

def get_center(part_map):
    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h, w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    return x_center, y_center


def sort_column(df):
    df = df.sort_values('file_name')
    return df


def evaluate(args, segmentation_module, train_dataset, test_dataset):
    trainloader = data.DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=False)
    testloader = data.DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)

    # put in eval mode
    segmentation_module.eval()

    size = args.input_size

    # iterate over train images to obtain predicted keypoints for train set
    print('Computing keypoints on train set. Please wait...')
    # obtain keypoint for train images
    out_df = {'file_name': [], 'value': []}
    lms_pred = []

    train_iter = iter(trainloader)
    with torch.no_grad():
        for _ in trange(len(trainloader.dataset)):

            batch = train_iter.next()

            image = batch['img']
            mask = batch['mask']
            name = batch['img_path'].split('/')[-1]

            # get the model output
            output = segmentation_module(image.to(args.gpu))[2]
            output = F.interpolate(output.cpu(), size=size, mode='bilinear')*mask

            centers = []
            for j in range(1, output.shape[1]):  # ignore the background
                part_map = output[0, j, ...] + 1e-6
                k = part_map.sum()
                part_map_pdf = part_map / k
                x_c, y_c = get_center(part_map_pdf)
                x_c = (x_c + 1.) / 2 * size[0]
                y_c = (y_c + 1.) / 2 * size[0]
                center = torch.stack((x_c, y_c), dim=0).unsqueeze(0)  # compute center of the part map
                centers.append(center)
            centers = torch.cat(centers, dim=0)
            lms_pred.append(centers.unsqueeze(0))

            out_df['value'].append(centers.numpy())
            out_df['file_name'].append(name)

        # save the landmarks in a pandas dataframe
        kp_train_df = pd.DataFrame(out_df)
        dataset_name_pkl = os.path.basename(opt.config).split('.')[0].split('-')[0]
        kp_train_df.to_pickle('landmarks/' + dataset_name_pkl + '_train_pred.pkl')

    # iterate over the test images to obtain predicted keypoints for test set
    print('Computing keypoints on test set. Please wait...')
    # obtain keypoint for test images
    out_df = {'file_name': [], 'value': []}
    lms_pred = []
    test_iter = iter(testloader)
    with torch.no_grad():
        for _ in trange(len(testloader.dataset)):
            batch = test_iter.next()

            image = batch['img']
            mask = batch['mask']
            name = batch['name'][0]

            # get the model output
            output = segmentation_module(image.to(args.gpu))[2]
            output = F.interpolate(output.cpu(), size=size, mode='bilinear')*mask

            centers = []
            for j in range(1, output.shape[1]):  # ignore the background
                part_map = output[0, j, ...] + 1e-6
                k = part_map.sum()
                part_map_pdf = part_map / k
                x_c, y_c = get_center(part_map_pdf)
                x_c = (x_c + 1.) / 2 * size[0]
                y_c = (y_c + 1.) / 2 * size[0]
                center = torch.stack((x_c, y_c), dim=0).unsqueeze(0)  # compute center of the part map
                centers.append(center)
            centers = torch.cat(centers, dim=0)
            lms_pred.append(centers.unsqueeze(0))

            out_df['value'].append(centers.numpy())
            out_df['file_name'].append(name)

        # save the landmarks in a pandas dataframe
        # kp_train_df = pd.DataFrame(out_df)
        # kp_train_df.to_pickle('landmarks/' + dataset_name_pkl + '_test_pred.pkl')

        # regress from predicted keypoints to ground truth landmarks
        df_kp = {}
        regress_keypoints(df_kp)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script")
    parser.add_argument("--config", required=True, help="path to the config file")
    parser.add_argument('--mode', default='evaluate')
    parser.add_argument("--root_dir", required=True, help="path to root folder of the train and test images")
    parser.add_argument("--checkpoint_path", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    segmentation_module = SegmentationModule(**config['model_params']['segmentation_module_params'],
                                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        segmentation_module.to(opt.device_ids[0])

    if opt.checkpoint_path is not None:
        checkpoint = torch.load(opt.checkpoint_path)
        load_segmentation_module(segmentation_module, checkpoint)

    dataset = {}
    dataset['train'] = FramesDataset(root_dir=opt.root_dir, is_train=True)
    dataset['test'] = FramesDataset(root_dir=opt.root_dir, is_train=False)

    evaluate(config, segmentation_module, dataset, opt)


