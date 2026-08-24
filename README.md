# README

# <span data-type="text" style="font-size: 19px;">UCMP: Uncertainty-Calibrated Masked Consistency and Prototype Distillation for Domain-Adaptive Semantic Segmentation</span>

# Abstract

Unsupervised domain-adaptive semantic segmentation aims to improve segmentation performance in target-domain scenarios by leveraging labeled source-domain data and unlabeled target-domain data. In recent years, teacher–student self-training and masked consistency learning have achieved substantial progress. However, although some studies have introduced pixel-level reliability estimation, masked consistency self-training frameworks represented by MIC still mainly rely on an image-level scalar to characterize pseudo-label quality, making it difficult to capture reliability variations among different pixels within the same image. Meanwhile, directly introducing pixel-level reweighting may alter the overall scale of target-domain supervision. In addition, existing masked consistency constraints are mainly imposed in the output space and lack explicit modeling of class-level feature consistency between full-views and masked-views. To address these issues, we propose UCMP, an uncertainty-calibrated masked consistency and prototype distillation method. First, normalized teacher prediction entropy is used to construct pixel-wise uncertainty weights, reducing the influence of highly uncertain pseudo-label on target-domain training. A mean-normalized weight calibration mechanism is then introduced to preserve relative reliability differences among pixels while maintaining the overall scale of target-domain supervision. Finally, masked prototype distillation is proposed to construct class prototypes from full-view teacher features and constrain masked-view student features to preserve consistent class semantics. Extensive experiments on GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes validate the effectiveness of the proposed method, achieving mIoU scores of 76.0% and 67.5%, respectively. All proposed modules are used only during training and introduce no additional inference complexity, thereby balancing target-domain adaptation performance and deployment efficiency.

# Dataset Link

 **·**  [GTA](https://download.visinf.tu-darmstadt.de/data/from_games/)

 **·**  [Cityscspes](https://www.cityscapes-dataset.com)

 **·**  [Synthia](https://synthia-dataset.net/downloads/)

# Training

We train the model by:

​`python run experiments.py --config configs/ucmp/gta2cs_ucmp_DAFormer.py`​

The entire installed python packages can be found in 'requirements.txt'.

# Ackonwledgement

This project is built upon previous projects. Especially, we'd like to thank the contributors of the following github repositories:

 **·**  [MIC](https://github.com/lhoyer/MIC)

 **·**  [HRDA](https://github.com/lhoyer/HRDA)

 **·**  [DAFormer](https://github.com/lhoyer/DAFormer)

 **·**  [MMsegmentation](https://github.com/open-mmlab/mmsegmentation)

 **·**  [SegFormer](https://github.com/NVlabs/SegFormer)

 **·**  [DACS](https://github.com/vikolss/DACS)

‍
