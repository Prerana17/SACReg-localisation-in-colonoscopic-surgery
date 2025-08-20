import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GeM(nn.Module):
    """
    Generalized Mean Pooling
    Args:
        p (float): initial p value
        eps (float): small value to avoid division by zero
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        # x: (N, C, H, W)
        # Pool across spatial dimensions (H, W)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class IRModel(nn.Module):
    def __init__(self, backbone_name='resnet50', agg_layer='gem'):
        super(IRModel, self).__init__()
        
        # 1. Backbone
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(pretrained=True)
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet50' or 'resnet101'.")
        
        # 移除 ResNet 的平均池化层和全连接层
        self.features = nn.Sequential(*list(backbone.children())[:-2]) 
        
        # 2. Aggregation Layer
        if agg_layer == 'gem':
            self.agg_layer = GeM()
            # 假设 ResNet-50/101 倒数第二层输出是 2048 维
            self.output_dim = 2048 
        elif agg_layer == 'netvlad':
            # NetVLAD 集成需要更多工作。你可以寻找 PyTorch 的 NetVLAD 实现库，例如：
            # pip install git+https://github.com/lyakaap/NetVLAD.git
            # from netvlad import NetVLAD
            # self.agg_layer = NetVLAD(num_clusters=64, dim=2048) # num_clusters 和 dim 需要根据你的数据调整
            # self.output_dim = 64 * 2048 # NetVLAD 输出维度
            raise NotImplementedError("NetVLAD integration requires a separate library or custom implementation.")
        else:
            raise ValueError("Unsupported aggregation layer. Choose 'gem' or 'netvlad'.")

    def forward(self, x):
        x = self.features(x)  # (N, C, H, W)
        x = self.agg_layer(x) # (N, C_out)
        # 展平特征向量
        x = x.view(x.size(0), -1) 
        # 归一化特征向量，使得它们位于单位球上
        x = F.normalize(x, p=2, dim=1) 
        return x
