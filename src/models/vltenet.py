import torch.nn as nn
import numpy as np
from .decoding import Decodeing
from . import hr
from . import resnet
from . import convnext
from . import unet_backbones
from . import vision_mamba
from . import densenet  # 新增导入
from . import efficientnet  # 新增导入


class Vltenet(nn.Module):
    def __init__(self, pretrained, final_kernel, dropout_rate=0.1, backbone='hrnet18',
                 use_transformer=False, transformer_config=None):
        super(Vltenet, self).__init__()

        self.backbone = backbone

        # 选择骨干网络
        if backbone == 'hrnet18':
            self.base_network = hr.hrnet18(pretrained=pretrained)
            self.use_multi_scale = True
        elif backbone == 'hrnet32':
            self.base_network = hr.hrnet32(pretrained=pretrained)
            self.use_multi_scale = True
        elif backbone == 'resnet50':
            self.base_network = resnet.resnet50(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'resnet101':
            self.base_network = resnet.resnet101(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'convnext_tiny':
            self.base_network = convnext.convnext_tiny_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'convnext_small':
            self.base_network = convnext.convnext_small_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'convnext_base':
            self.base_network = convnext.convnext_base_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'unet':
            self.base_network = unet_backbones.unet_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'unetplusplus':
            self.base_network = unet_backbones.unetplusplus_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'vision_mamba':
            self.base_network = vision_mamba.vision_mamba_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        # 新增DenseNet支持
        elif backbone == 'densenet121':
            self.base_network = densenet.densenet121(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'densenet169':
            self.base_network = densenet.densenet169(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'densenet201':
            self.base_network = densenet.densenet201(pretrained=pretrained)
            self.use_multi_scale = False
        # 新增EfficientNet支持
        elif backbone == 'efficientnet_b0':
            self.base_network = efficientnet.efficientnet_b0_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'efficientnet_b3':
            self.base_network = efficientnet.efficientnet_b3_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        elif backbone == 'efficientnet_b5':
            self.base_network = efficientnet.efficientnet_b5_backbone(pretrained=pretrained)
            self.use_multi_scale = False
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 默认的Transformer配置
        default_transformer_config = {
            'depth': 4,
            'num_heads': 4,
            'window_height': 16,
            'window_width': 16,
            'shift_size': None,
            'channel_expansion': 2,
            'mlp_ratio': 4,
            'drop_rate': dropout_rate,
            'attn_drop_rate': dropout_rate,
            'downsample_factor': 4  # 默认降采样因子
        }

        # 合并用户配置
        if transformer_config is not None:
            for key, value in transformer_config.items():
                default_transformer_config[key] = value

        self.Decodeing = Decodeing(
            final_kernel=final_kernel,
            head_conv=256,
            channel=64,
            use_gnn=False,
            use_trans=use_transformer,  # 根据参数决定是否使用Transformer
            dropout_rate=dropout_rate,
            backbone=backbone,
            trans_config=default_transformer_config
        )

    def forward(self, x):
        if self.use_multi_scale:
            # HRNet返回多尺度特征
            features = self.base_network(x)
            feature_dict = self.Decodeing(features)
        else:
            # 其他backbone只返回单一特征
            feature = self.base_network(x)
            feature_dict = self.Decodeing(feature)

        return feature_dict