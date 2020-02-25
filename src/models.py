import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        self.grapheme_root = nn.Linear(512, 168)
        self.vowel_diacritic = nn.Linear(512, 11)
        self.consonant_diacritic = nn.Linear(512, 7)

        def forward(self, X):
            batch_size, _, _, _ = X.shape
            X = self.model.features(X)
            X = F.adaptive_avg_pool2d(X, 1).reshape(batch_size, -1)
            grapheme_root = self.grapheme_root(X)
            vowel_diacritic = self.vowel_diacritic(X)
            consonant_ diacritic = self.consonant_diacritic(X)

            return grapheme_root, vowel_diacritic, consonant_diacritic