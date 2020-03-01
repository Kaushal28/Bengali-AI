import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

def custom_load_pretrained_weights(model, model_name, load_fc=True, advprop=False):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    # AutoAugment or Advprop (different preprocessing)
    url_map_ = url_map_advprop if advprop else url_map
    state_dict = torch.load('efficientnet-b1-dbc7070a.pth', map_location='cpu')
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))

from efficientnet_pytorch import utils
utils.load_pretrained_weights.__code__ = custom_load_pretrained_weights.__code__


class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2


class EfficientNetWrapper(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetWrapper, self).__init__()
        
        # Load imagenet pre-trained model 
        self.effNet = EfficientNet.from_pretrained('efficientnet-b1', in_channels=3)
        
        # Appdend output layers based on our date
        self.fc_root = nn.Linear(in_features=1000, out_features=168)
        self.fc_vowel = nn.Linear(in_features=1000, out_features=11)
        self.fc_consonant = nn.Linear(in_features=1000, out_features=7)
        
    def forward(self, X):
        output = self.effNet(X)
        output_root = self.fc_root(output)
        output_vowel = self.fc_vowel(output)
        output_consonant = self.fc_consonant(output)
        
        return output_root, output_vowel, output_consonant