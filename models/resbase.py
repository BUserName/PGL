from __future__ import absolute_import
import torch.nn as nn
from torchvision import models

__all__ = ["RESBase"]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RESBase(nn.Module):
    def __init__(self, args):
        super(RESBase, self).__init__()
        self.args = args
        self.CNN = models.resnet50(pretrained=True)
        modules = list(self.CNN.children())[:-1]
        self.CNN = nn.Sequential(*modules)
        #
        # modules = list(res.classifier.children())[:-1]
        # self.extractor = nn.Sequential(
        #
        #     nn.Linear(2048,2048),
        #     nn.BatchNorm1d(2048)
        # )

    def forward(self, inputs):
        assert len(inputs.shape) == 5
        batch_size, num_sample, channel, width, height = inputs.size()
        outputs = self.CNN(inputs.view(-1, channel, width, height))
        outputs = outputs.view(batch_size, num_sample, -1)
        # features = self.extractor(outputs)
        # features = features.view(batch_size, num_sample, -1)
        return outputs


