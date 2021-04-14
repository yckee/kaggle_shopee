import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
 
from efficientnet_pytorch import EfficientNet


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5, eps=1e-6):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.threshold = math.pi - margin
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            return cos_theta

        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        mask = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot)

        logits = torch.cos(torch.where(mask.bool(), theta + self.margin, theta))
        logits *= self.scale

        return logits
        

class ShopeeNet(nn.Module):
    def __init__(self, feature_space, out_features, scale, margin):
        super(ShopeeNet, self).__init__()
        self.feature_space = feature_space
        self.out_features = out_features
        
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.backbone._conv_head.out_channels
        self.dropout = nn.Dropout(p=self.backbone._global_params.dropout_rate)
        self.classifier = nn.Linear(in_features, self.feature_space)
        self.bn = nn.BatchNorm1d(self.feature_space)
        
        self.margin = ArcFace(
            in_features = self.feature_space,
            out_features = self.out_features,
            scale = scale, 
            margin = margin       
        )
        
        if self.training:
            self._init_params()
        

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


    def forward(self, img, label=None):
        batch_size = img.shape[0]
        out = self.backbone.extract_features(img)
        out = self.backbone._avg_pooling(out).view(batch_size, -1)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.bn(out) 
        
        if self.training:
            logits = self.margin(out, label)
            return logits
        else:
            logits = self.margin(out)
            return logits