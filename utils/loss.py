import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, output, target):

        input_soft = (output + 1e-6).float()
        batch_size, num_class = output.shape

        # create the labels one hot tensor
        target_one_hot = torch.FloatTensor(batch_size, num_class).cuda()
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target.view(-1, 1), 1)

        # compute the actual focal loss
        weight = torch.pow(torch.tensor(1.) - input_soft, self.gamma).float()

        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss
