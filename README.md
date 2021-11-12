## Unsupervised Part Discovery from Contrastive Reconstruction
#### Subhabrata Choudhury, Iro Laina, Christian Rupprecht, Andrea Vedaldi
[![Conference](https://img.shields.io/badge/NeurIPS-2021-purple?style=flat-square&color=f1e3ff&labelColor=69448c)](https://nips.cc/Conferences/2021/ScheduleMultitrack?event=26254)  

### Code will be relased soon.


##### Abstract:
> <sup> The goal of self-supervised visual representation learning is to learn strong, transferable image representations, with the majority of research focusing on object or scene level. On the other hand, representation learning at part level has received significantly less attention. In this paper, we propose an unsupervised approach to object part discovery and segmentation and make three contributions. First, we construct a proxy task through a set of objectives that encourages the model to learn a meaningful decomposition of the image into its parts. Secondly, prior work argues for reconstructing or clustering pre-computed features as a proxy to parts; we show empirically that this alone is unlikely to find meaningful parts; mainly because of their low resolution and the tendency of classification networks to spatially smear out information. We suggest that image reconstruction at the level of pixels can alleviate this problem, acting as a complementary cue. Lastly, we show that the standard evaluation based on keypoint regression does not correlate well with segmentation quality and thus introduce different metrics, NMI and ARI, that better characterize the decomposition of objects into parts. Our method yields semantic parts which are consistent across fine-grained but visually distinct categories, outperforming the state of the art on three benchmark datasets. Code is available at the project page: https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/. </sup>


### Citation   
```
@inproceedings{choudhury21unsupervised,
 author = {Subhabrata Choudhury and Iro Laina and Christian Rupprecht and Andrea Vedaldi},
 booktitle = {Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
 title = {Unsupervised Part Discovery from Contrastive Reconstruction},
 year = {2021}
}
```   
