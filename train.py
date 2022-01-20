"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import copy
import glob
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2

# solve potential deadlock https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)

import hydra
import torch.backends.cudnn as cudnn
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets.lit_dataset import LitDataset
from utils.utils import seed_worker
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    print(os.getcwd())
    cfg = dict(cfg)
    if cfg['dataset_name'] is not None:
        cfg['dataset'] = cfg['dataset_name']
    wandb.init(project='unsup-parts', entity='wandb_userid')
    wandb.config.update(cfg)
    args = wandb.config
    cudnn.enabled = True

    if args.exp_name is not None:
        api = wandb.Api()
        run = api.run(path=f'wandb_userid/unsup-parts/{wandb.run.id}')
        run.name = f'{args.exp_name}-{run.name}'
        run.save()

    print("---------------------------------------")
    print(f"Arguments received: ")
    print("---------------------------------------")
    for k, v in sorted(args.__dict__.items()):
        print(f"{k:25}: {v}")
    print("---------------------------------------")

    code_path = os.path.join(os.path.join(wandb.run.dir, 'files', 'full_code'))
    Path(code_path).mkdir(parents=True, exist_ok=True)
    for file in glob.glob('*.py'):
        shutil.copy(file, code_path)
    shutil.copytree('datasets', code_path + '/datasets', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('configs', code_path + '/configs', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('tps', code_path + '/tps', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('tps_stn_pytorch', code_path + '/tps_stn_pytorch', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('models', code_path + '/models', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('utils', code_path + '/utils', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    Path(osp.join(wandb.run.dir.replace('/wandb/', '/outputs/'), 'files')).mkdir(parents=True, exist_ok=True)

    if args.dataset == 'CUB':
        from datasets import cub
        args.split = 'train'
        train_dataset = cub.CUBDataset(args)
        args_test = SimpleNamespace(**copy.deepcopy(dict(args)))
        args_test.split = 'test'
        test_dataset = cub.CUBDataset(args_test)

    elif args.dataset == 'DF':
        from datasets import deepfashion
        args.split = 'train'
        train_dataset = deepfashion.DFDataset(args)
        args_test = SimpleNamespace(**copy.deepcopy(dict(args)))
        args_test.split = 'test'
        test_dataset = deepfashion.DFDataset(args_test)

    elif args.dataset == 'PP':
        from datasets import pascal_parts
        args.split = 'train'
        train_dataset = pascal_parts.PPDataset(args)
        args_test = SimpleNamespace(**copy.deepcopy(dict(args)))
        args_test.split = 'test'
        test_dataset = pascal_parts.PPDataset(args_test)

    else:
        print(f'Invalid dataset {args.dataset}')
        sys.exit()

    trainloader = DataLoader(
        LitDataset(train_dataset, args.use_lab),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        drop_last=True)

    trainloader_iter = enumerate(trainloader)

    testloader = DataLoader(
        LitDataset(test_dataset),
        batch_size=args_test.batch_size,
        shuffle=True,
        num_workers=args_test.num_workers,
        worker_init_fn=seed_worker,
        drop_last=False)

    testloader_iter = enumerate(testloader)

    # Initialize trainer
    from trainer_ours import Trainer
    trainer = Trainer(args)

    for i_iter in range(args.num_steps):

        try:
            _, batch = trainloader_iter.__next__()
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()

        trainer.step(batch, i_iter)

        if i_iter % args.vis_interval == 0 or i_iter == 100:
            trainer.visualize('train', i_iter, batch)
            try:
                _, test_batch = testloader_iter.__next__()
            except:
                testloader_iter = enumerate(testloader)
                _, test_batch = testloader_iter.__next__()

            if test_batch['img'].size(0) == args.batch_size:
                trainer.visualize('test', i_iter, test_batch)

        # can be used for tracking purposes but for final numbers please use the script in the evaluation folder
        # if i_iter % 500 == 0 and args.dataset == 'CUB':
        #     trainer.log_nmi(train_dataset, test_dataset, i_iter)
        # if (i_iter % 5000 == 0 or i_iter == 1000) and i_iter > 0 and args.dataset in ['PP', 'DF']:
        #     trainer.log_ari(testloader, i_iter)

        if i_iter >= args.num_steps - 1:
            print('save model ...')
            trainer.save_model(osp.join(wandb.run.dir.replace('/wandb/', '/outputs/'), 'model_' + str(args.num_steps) + '.pth'), i_iter)
            break
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            trainer.save_model(osp.join(wandb.run.dir.replace('/wandb/', '/outputs/'), 'model_' + str(i_iter) + '.pth'), i_iter)


if __name__ == '__main__':
    main()
