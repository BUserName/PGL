import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function


# class GradReverse(Function):
#     def __init__(self, lambd):
#         self.lambd = lambd
#
#     def forward(self, x):
#         return x.view_as(x)
#
#     def backward(self, grad_output):
#         return (grad_output * -self.lambd)
# def grad_reverse(x, lambd=1.0):
#     return GradReverse(lambd)(x)

class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        x_out = x_out.squeeze(-1)
        return x_out

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.classifier = nn.Sequential(
            nn.Linear(2048, self.args.num_class-1)
        )
    def forward(self, inputs):
        return self.classifier(inputs)

