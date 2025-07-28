import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def resnet50(pretrained=True):
    """
    创建预训练的ResNet50模型，并返回单一特征图

    Args:
        pretrained: 是否使用预训练权重

    Returns:
        修改后的ResNet50模型
    """
    model = models.resnet50(pretrained=pretrained)
    return ResNetFeatures(model)


def resnet101(pretrained=True):
    """
    创建预训练的ResNet101模型，并返回单一特征图

    Args:
        pretrained: 是否使用预训练权重

    Returns:
        修改后的ResNet101模型
    """
    model = models.resnet101(pretrained=pretrained)
    return ResNetFeatures(model)


class ResNetFeatures(nn.Module):
    """
    从ResNet获取单一特征图的封装器，并支持上采样到所需分辨率
    """

    def __init__(self, model, output_stride=16):
        super(ResNetFeatures, self).__init__()
        self.output_stride = output_stride

        # 保留原始ResNet的层
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1  # 1/4
        self.layer2 = model.layer2  # 1/8
        self.layer3 = model.layer3  # 1/16

        # 根据所需输出步长调整layer4
        if output_stride == 16:
            # 修改layer4的步长为1以保持1/16分辨率
            for n, m in model.layer4.named_modules():
                if 'conv2' in n or 'downsample.0' in n:
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
            self.layer4 = model.layer4
        elif output_stride == 8:
            # 修改layer3和layer4的步长为1以保持1/8分辨率
            for n, m in model.layer3.named_modules():
                if 'conv2' in n or 'downsample.0' in n:
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
            self.layer3 = model.layer3

            for n, m in model.layer4.named_modules():
                if 'conv2' in n or 'downsample.0' in n:
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
            self.layer4 = model.layer4
        else:
            # 默认保持原始步长
            self.layer4 = model.layer4

        # 降维卷积
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        # 标准ResNet前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 降维到所需通道数
        x = self.reduce_dim(x)

        # 计算需要的上采样因子
        # 对于典型的ResNet，此时特征图尺寸为输入的1/32或1/16(取决于output_stride)
        current_stride = 32 if self.output_stride == 32 else 16

        # 上采样至1/4原始分辨率(与HRNet输出匹配)
        if current_stride > 4:
            upscale_factor = current_stride // 4
            x = F.interpolate(
                x,
                scale_factor=upscale_factor,
                mode='bilinear',
                align_corners=False
            )

        # 返回单一特征图
        return x