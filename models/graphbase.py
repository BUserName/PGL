from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.init as init
from torchvision import models
# from models.__init__ import weight_init

__all__ = ["GraphNetwork"]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 2,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x node_size x node_size) only use inter dist.
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).cuda()

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat.squeeze(1), node_feat)

        node_feat = torch.cat([node_feat, aggr_feat], -1).transpose(1, 2)
        # node_feat = ((0*node_feat + 2*aggr_feat)/2).transpose(1,2)

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        # for m in self.network.children():
        #     return m(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze(1)

        # diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).cuda()
        # edge_feat = edge_feat * diag_mask
        # merge_sum = torch.sum(edge_feat, -1, True)
        # # set diagonal as zero and normalize
        # edge_feat = F.normalize(sim_val * edge_feat, p=1, dim=-1) * merge_sum
        force_edge_feat = torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).cuda()
        edge_feat = sim_val + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1)

        return edge_feat

class GraphNetwork(nn.Module):
    def __init__(self, args):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = args.in_features
        self.node_features = args.node_features
        self.edge_features = args.edge_features
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        # for each layer
        for l in range(self.num_layers):
            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers - 1 else 0.0)

            # set edge to node
            edge2node_net = NodeUpdateNetwork(
                in_features=self.in_features if l == 0 else self.node_features,
                num_features=self.node_features if l != self.num_layers - 1 else args.num_class - 1,
                dropout=self.dropout if l < self.num_layers - 1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, init_node_feat, init_edge_feat, target_mask):
        # for each layer
        edge_feat_list = []
        node_feat_list = []
        node_feat = init_node_feat
        edge_feat = init_edge_feat

        for l in range(self.num_layers):
            # (1) edge update
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)

            # take similarity only for unk
            # inverse_mask = ~target_mask.type(torch.ByteTensor)
            # combined_edge_feat = target_mask.type(torch.FloatTensor).cuda() * edge_feat + \
            #                      inverse_mask.type(torch.FloatTensor).cuda() * init_edge_feat

            # (2) node update
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # save edge feature
            edge_feat_list.append(edge_feat)
            node_feat_list.append(node_feat)

        return edge_feat_list, node_feat_list