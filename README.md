## Unsupervised Part Discovery from Contrastive Reconstruction
#### Subhabrata Choudhury, Iro Laina, Christian Rupprecht, Andrea Vedaldi
### [![ProjectPage](https://img.shields.io/badge/-Project%20Page-magenta.svg?style=for-the-badge&color=white&labelColor=magenta)](https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/) [![Conference](https://img.shields.io/badge/NeurIPS-2021-purple.svg?style=for-the-badge&color=f1e3ff&labelColor=purple)](https://nips.cc/Conferences/2021/Schedule?showEvent=26254)    [![arXiv](https://img.shields.io/badge/arXiv-2111.06349-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2111.06349)

### Setup

```shell
git clone https://github.com/subhc/unsup-parts.git
cd unsup-parts
conda env create --file environment.yml
conda activate unsup-parts
wget https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/files/checkpoints.tar.gz
tar zxvf checkpoints.tar.gz
```
The project uses Weights & Biases for visualization, please update `wandb_userid` in `train.py` to your username
### Data Preparation:

#### CUB-200-2011

1. Download [CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and [segmentations.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz) from the [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) provided links.
2. Download [cachedir.tar.gz](https://www.dropbox.com/sh/ea3yprgrcjuzse5/AAB476Nn0Lwbrt3iuedB9yzIa?dl=0) mentioned [here](https://github.com/akanazawa/cmr/issues/3#issuecomment-451757610).
3. Create a directory named `data` with the following folder structure inside and extract the tars at the mentioned locations.
4. Train a segmentation network to predict foreground masks for the test split, or download precalculated outputs: [cub_supervisedlabels.tar.gz](https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/files/cub_supervisedlabels.tar.gz) (17MB).

```
data
└── CUB  # extract CUB_200_2011.tgz, cub_supervisedlabels.tar.gz here
    ├── CUB_200_2011 # extract cachedir.tar.gz and segmentations.tgz here       
    │   ├── attributes
    │   ├── cachedir
    │   ├── images
    │   ├── parts
    │   └── segmentations
    └── supervisedlabels
```
Example
```shell
mkdir -p data/CUB/
cd data/CUB/
tar zxvf CUB_200_2011.tgz 
tar zxvf cub_supervised_labels.tar.gz 
cd CUB_200_2011
tar zxvf segmentations.tgz
tar zxvf cachedir.tar.gz

```


#### DeepFashion
1. Create a directory named `data` with the folder structure below.
2. Download the [segmentation](https://drive.google.com/drive/folders/1X6FjyMyJmFLKs3M8RyY8zioKEfE2nvmm) folder from the [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) provided links.
3. Extract img_highres_seg.zip inside `segmentation` Folder.
4. Train a segmentation network to predict foreground masks for the test split, or download precalculated outputs: [deepfashion_supervisedlabels.tar.gz](https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/files/deepfashion_supervisedlabels.tar.gz) (56MB).
```
data
└── DeepFashion
    └── In-shop Clothes Retrieval Benchmark  # extract deepfashion_supervisedlabels.tar.gz here
        ├── Anno  
        │   └── segmentation # extract img_highres_seg.zip here
        │       └── img_highres
        │           ├── MEN
        │           └── WOMEN
        └── supervisedlabels
            └── img_highres
                ├── MEN
                └── WOMEN
```
Example
```shell
mkdir -p data/DeepFashion/In-shop\ Clothes\ Retrieval\ Benchmark/Anno/
cd data/DeepFashion/In-shop\ Clothes\ Retrieval\ Benchmark/
wget https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/files/deepfashion_supervisedlabels.tar.gz
tar zxvf deepfashion_supervisedlabels.tar.gz
cd Anno
# get the segmentation folder from the google drive link
cd segmentation
unzip img_highres_seg.zip
```
### Training:
To train CUB:
```shell
python train.py dataset_name=CUB
```
To train DeepFashion:
```shell
python train.py dataset_name=DF
```

### Evaluation:
You can find evaluation code in the [evaluation folder](evaluation).

### Pretrained weights:

| Description | Size | Link |
| ---- | -----| ----|
| CUB-200-2011 (pth) | 181MB | [here](https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/files/checkpoints/CUB/model_60000.pth) |
| DeepFashion (pth) | 181MB| [here](https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/files/checkpoints/DeepFashion/model_100000.pth) |
| Both (tar.gz) | 351MB| [here](https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/files/checkpoints.tar.gz) |

Please move the `pth` files in the `checkpoints/CUB` and `checkpoints/DeepFashion` folders respectively. 


### Abstract:
<sup> The goal of self-supervised visual representation learning is to learn strong, transferable image representations, with the majority of research focusing on object or scene level. On the other hand, representation learning at part level has received significantly less attention. In this paper, we propose an unsupervised approach to object part discovery and segmentation and make three contributions. First, we construct a proxy task through a set of objectives that encourages the model to learn a meaningful decomposition of the image into its parts. Secondly, prior work argues for reconstructing or clustering pre-computed features as a proxy to parts; we show empirically that this alone is unlikely to find meaningful parts; mainly because of their low resolution and the tendency of classification networks to spatially smear out information. We suggest that image reconstruction at the level of pixels can alleviate this problem, acting as a complementary cue. Lastly, we show that the standard evaluation based on keypoint regression does not correlate well with segmentation quality and thus introduce different metrics, NMI and ARI, that better characterize the decomposition of objects into parts. Our method yields semantic parts which are consistent across fine-grained but visually distinct categories, outperforming the state of the art on three benchmark datasets. Code is available at the [project page](https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/). </sup>

### Citation   
```
@inproceedings{choudhury21unsupervised,
 author = {Subhabrata Choudhury and Iro Laina and Christian Rupprecht and Andrea Vedaldi},
 booktitle = {Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
 title = {Unsupervised Part Discovery from Contrastive Reconstruction},
 year = {2021}
}
```   
### Acknowledgement
Code is largely based on [SCOPS](https://github.com/NVlabs/SCOPS).
