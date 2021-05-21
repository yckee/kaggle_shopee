import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
 
import timm
import transformers
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
    def __init__(self, model_name, feature_space, out_features, scale, margin):
        super(ShopeeNet, self).__init__()
        self.feature_space = feature_space
        self.out_features = out_features
        
        if 'efficientnet' in model_name:
            self.backbone = EfficientNet.from_pretrained(model_name)
            self.in_features = self.backbone._conv_head.out_channels
            self.timm = False
        
        elif 'nfnet' in model_name:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.in_features = self.backbone.final_conv.out_channels
            self.timm = True
        
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.in_features, self.feature_space)
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
        features = self.extract_features(img)
        
        if self.training:
            logits = self.margin(features, label)
            return logits
        else:
            logits = self.margin(features)
            return logits

    def extract_features(self, x):    
        batch_size = x.shape[0]

        if self.timm:
            x = self.backbone(x).view(batch_size, -1)

        else:
            x = self.backbone.extract_features(x)
            x = self.backbone._avg_pooling(x).view(batch_size, -1)

        x = self.dropout(x)
        x = self.classifier(x)
        x = self.bn(x) 
        
        return x



class ShopeeBert(nn.Module):

    def __init__(self, model_name, feature_space, out_features, scale, margin):
        super(ShopeeBert, self).__init__()
        self.feature_space = feature_space
        self.out_features = out_features

        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.in_features = self.transformer.config.hidden_size
        
        
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.in_features, self.feature_space)
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


    def forward(self, input_ids, attention_mask, label=None):
        features = self.extract_text_feat(input_ids, attention_mask)

        if self.training:
            logits = self.margin(features, label)
            return logits
        else:
            logits = self.margin(features)
            return logits
    
    
    def extract_text_feat(self, input_ids, attention_mask):
        # inputs = self.tokenizer(text, max_length=128, padding='max_length', truncation=True,  return_tensors='pt')
        x = self.transformer(input_ids, attention_mask)

        features = x[0]
        features = features[:, 0, :]

        features = self.dropout(features)
        features = self.classifier(features)
        features = self.bn(features) 
        
        return features        


class ShopeeCombined(nn.Module):

    def __init__(self, cnn_name, bert_name,  feature_space, out_features, scale, margin):
        super(ShopeeCombined, self).__init__()
        self.feature_space = feature_space
        self.out_features = out_features

        
        self.backbone = timm.create_model(cnn_name, pretrained=True, num_classes=0)
        
        self.transformer = transformers.AutoModel.from_pretrained(bert_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(bert_name)
        
        self.in_features = self.transformer.config.hidden_size + self.backbone.final_conv.out_channels

        
        
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.in_features, self.feature_space)
        self.bn = nn.BatchNorm1d(self.feature_space)
        
        self.margin = ArcFace(
            in_features = self.feature_space,
            out_features = self.out_features,
            scale = scale, 
            margin = margin       
        )
        
        if self.training:
            self._init_params()

        self.saving_name = f"{bert_name.split('-')[0]}_{cnn_name.split('_')[-1]}_fc{feature_space}_s{scale}_m{str(margin).replace(',', '')}"


    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


    def forward(self, img, input_ids, attention_mask, label=None):
        features = self.extract_features(img, input_ids, attention_mask)

        if self.training:
            logits = self.margin(features, label)
            return logits
        else:
            logits = self.margin(features)
            return logits


    def extract_features(self, img, input_ids, attention_mask): 
        batch_size = img.shape[0]

        img_feat = self.backbone(img).view(batch_size, -1)
        
        text_feat = self.transformer(input_ids, attention_mask)
        text_feat = text_feat[0]
        text_feat = text_feat[:, 0, :]

        feat = torch.cat([img_feat, text_feat], dim=1)

        feat = self.dropout(feat)
        feat = self.classifier(feat)
        feat = self.bn(feat) 
        
        return feat