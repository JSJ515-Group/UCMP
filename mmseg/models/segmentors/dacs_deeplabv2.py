import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import SEGMENTORS
from .base import BaseSegmentor


AFFINE_PAR = True


def conv3x3(
        in_planes,
        out_planes,
        stride=1):

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            dilation=1,
            downsample=None):

        super().__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=stride,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            planes,
            affine=AFFINE_PAR
        )

        self._freeze_bn_affine(
            self.bn1
        )

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm2d(
            planes,
            affine=AFFINE_PAR
        )

        self._freeze_bn_affine(
            self.bn2
        )

        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion,
            affine=AFFINE_PAR
        )

        self._freeze_bn_affine(
            self.bn3
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.downsample = downsample
        self.stride = stride

    @staticmethod
    def _freeze_bn_affine(bn):

        for parameter in bn.parameters():
            parameter.requires_grad = False

    def forward(self, x):

        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output


class ClassifierModule(nn.Module):
    """
    原始DACS DeepLabV2分类器。

    对应四个并行3×3空洞卷积分支：

        dilation = 6, 12, 18, 24

    四个分支的logits直接相加。
    """

    def __init__(
            self,
            dilation_series,
            padding_series,
            num_classes):

        super().__init__()

        self.conv2d_list = nn.ModuleList()

        for dilation, padding in zip(
                dilation_series,
                padding_series):

            self.conv2d_list.append(
                nn.Conv2d(
                    2048,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True
                )
            )

    def forward(self, x):

        output = self.conv2d_list[0](x)

        for index in range(
                1,
                len(self.conv2d_list)):

            output = (
                output
                + self.conv2d_list[index](x)
            )

        return output


@SEGMENTORS.register_module()
class DACSDeepLabV2(BaseSegmentor):
    """
    与vikolss/DACS原始Res_Deeplab参数结构兼容的
    MMSegmentation Segmentor。

    权重键保持为：

        conv1.*
        bn1.*
        layer1.*
        layer2.*
        layer3.*
        layer4.*
        layer5.conv2d_list.*

    因此可直接加载原始checkpoint['model']。
    """

    def __init__(
            self,
            num_classes=19,
            train_cfg=None,
            test_cfg=None,
            pretrained=None):

        super().__init__()

        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            64,
            affine=AFFINE_PAR
        )

        self._freeze_bn_affine(
            self.bn1
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=True
        )

        # ResNet-101: [3, 4, 23, 3]
        self.layer1 = self._make_layer(
            Bottleneck,
            planes=64,
            blocks=3
        )

        self.layer2 = self._make_layer(
            Bottleneck,
            planes=128,
            blocks=4,
            stride=2
        )

        self.layer3 = self._make_layer(
            Bottleneck,
            planes=256,
            blocks=23,
            stride=1,
            dilation=2
        )

        self.layer4 = self._make_layer(
            Bottleneck,
            planes=512,
            blocks=3,
            stride=1,
            dilation=4
        )

        self.layer5 = ClassifierModule(
            dilation_series=[
                6,
                12,
                18,
                24
            ],
            padding_series=[
                6,
                12,
                18,
                24
            ],
            num_classes=num_classes
        )

        self._initialize_weights()

    @staticmethod
    def _freeze_bn_affine(bn):

        for parameter in bn.parameters():
            parameter.requires_grad = False

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,
            dilation=1):

        downsample = None

        output_channels = (
            planes
            * block.expansion
        )

        if (
            stride != 1
            or self.inplanes != output_channels
            or dilation in (2, 4)
        ):

            downsample_bn = nn.BatchNorm2d(
                output_channels,
                affine=AFFINE_PAR
            )

            self._freeze_bn_affine(
                downsample_bn
            )

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                downsample_bn
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample
            )
        ]

        self.inplanes = output_channels

        for _ in range(1, blocks):

            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=dilation
                )
            )

        return nn.Sequential(
            *layers
        )

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                    module,
                    nn.Conv2d):

                kernel_height = (
                    module.kernel_size[0]
                )

                kernel_width = (
                    module.kernel_size[1]
                )

                n = (
                    kernel_height
                    * kernel_width
                    * module.out_channels
                )

                module.weight.data.normal_(
                    0,
                    np.sqrt(2.0 / n)
                )

                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(
                    module,
                    nn.BatchNorm2d):

                if module.weight is not None:
                    module.weight.data.fill_(1)

                if module.bias is not None:
                    module.bias.data.zero_()

        # 原始DeepLabV2分类器使用较小初始化
        for classifier in (
                self.layer5.conv2d_list):

            classifier.weight.data.normal_(
                0,
                0.01
            )

            if classifier.bias is not None:
                classifier.bias.data.zero_()

    def extract_feat(self, img):
        """
        提取layer4输出。
        """

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def encode_decode(
            self,
            img,
            img_metas):

        feature = self.extract_feat(
            img
        )

        logits = self.layer5(
            feature
        )

        # 原始DACS评估使用align_corners=True
        logits = F.interpolate(
            logits,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        return logits

    def forward_train(
            self,
            img,
            img_metas,
            gt_semantic_seg,
            **kwargs):

        raise NotImplementedError(
            '该适配器仅用于加载原始DACS权重并执行推理。'
        )

    def simple_test(
            self,
            img,
            img_meta,
            rescale=True):

        logits = self.encode_decode(
            img,
            img_meta
        )

        prediction = torch.argmax(
            logits,
            dim=1
        )

        results = []

        for batch_index in range(
                prediction.shape[0]):

            current_prediction = prediction[
                batch_index:
                batch_index + 1
            ].unsqueeze(1).float()

            if rescale:

                ori_height = int(
                    img_meta[
                        batch_index
                    ]['ori_shape'][0]
                )

                ori_width = int(
                    img_meta[
                        batch_index
                    ]['ori_shape'][1]
                )

                # 原始DACS先在512×1024上argmax。
                # 为了和原始Cityscapes图片排版，
                # 再用最近邻恢复到原图尺寸。
                current_prediction = (
                    F.interpolate(
                        current_prediction,
                        size=(
                            ori_height,
                            ori_width
                        ),
                        mode='nearest'
                    )
                )

            current_prediction = (
                current_prediction[
                    0,
                    0
                ]
                .long()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            results.append(
                current_prediction
            )

        return results

    def aug_test(
            self,
            imgs,
            img_metas,
            rescale=True):

        if len(imgs) != 1:

            raise NotImplementedError(
                'DACS适配器只支持单尺度、无翻转推理。'
            )

        return self.simple_test(
            imgs[0],
            img_metas[0],
            rescale=rescale
        )