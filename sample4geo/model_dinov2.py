# sample4geo/model_dinov2.py
import torch
import torch.nn as nn
import math
import numpy as np


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=16, alpha=32):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 初始化低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros((original_linear.in_features, r)))
        self.lora_B = nn.Parameter(torch.zeros((r, original_linear.out_features)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 冻结原始全连接层的计算 + LoRA旁路计算
        return self.original_linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def inject_lora(model, r=8, alpha=8):
    """遍历模型，将注意力机制中的 qkv 投影层替换为 LoRA 层"""
    for name, module in model.named_modules():
        if 'qkv' in name and isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, LoRALinear(module, r, alpha))


class DINOv2_Geo(nn.Module):
    def __init__(self, pretrained_path, r=8):
        super().__init__()
        # 加载 DINOv2 架构 (vitb14 输出维度为 768)
        self.backbone = torch.hub.load('./facebookresearch_dinov2_main/dinov2', 'dinov2_vitb14', source='local', pretrained=False)

        # 加载本地预训练权重
        state_dict = torch.load(pretrained_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=True)

        # 冻结全部主干参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 注入 LoRA (仅对 qkv 计算梯度)
        inject_lora(self.backbone, r=r, alpha=r)

        self.embed_dim = 768  # DINOv2-B 的特征维度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x1, x2=None):
        # DINOv2 直接输出的就是 CLS token，用于全局特征检索
        f1 = self.backbone(x1)
        if x2 is not None:
            f2 = self.backbone(x2)
            return f1, f2
        return f1
