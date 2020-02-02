from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
# from models.__init__ import weight_init

__all__ = ["VGGBase"]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VGGBase(nn.Module):
    def __init__(self, args):
        super(VGGBase, self).__init__()
        self.args = args
        vgg = models.vgg19(pretrained=True)
        # modules = list(vgg.features.children())
        self.CNN = vgg.features
        modules = list(vgg.classifier.children())[:-1]
        self.extractor = nn.Sequential(
            vgg.avgpool,
            Flatten(),
            *modules,
            nn.BatchNorm1d(4096)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(4096, 1024),
        #     nn.BatchNorm1d(1024),
        #
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #
        #
        #     nn.Linear(1024, args.num_class)
        # )

    # nn.Dropout(args.dropout),
    # nn.LeakyReLU(negative_slope=0.2, inplace=True),
    # nn.LeakyReLU(negative_slope=0.2, inplace=True),

    def forward(self, inputs):
        assert len(inputs.shape) == 5
        batch_size, num_sample, channel, width, height = inputs.size()
        outputs = self.CNN(inputs.view(-1, channel, width, height))
        features = self.extractor(outputs)
        # outputs = self.classifier(features)
        features = features.view(batch_size, num_sample, -1)
        # outputs = outputs.view(batch_size, num_sample, -1)

        return features


