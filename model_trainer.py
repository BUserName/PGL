import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import models
from models.__init__ import weight_init
from torchvision import transforms
from torch.utils.data import DataLoader
import utils

import os.path as osp
from tqdm import tqdm
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from utils.logger import AverageMeter as meter
from data_loader import Visda_Dataset, Office_Dataset, Home_Dataset, Visda18_Dataset
from utils.loss import FocalLoss

from models.component import Discriminator, Classifier
###heat M
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

class ModelTrainer():
    def __init__(self, args, data, step=0, label_flag=None, v=None, logger=None):
        self.args = args
        self.batch_size = args.batch_size
        self.data_workers = 6

        self.step = step
        self.data = data
        self.label_flag = label_flag

        self.num_class = data.num_class
        self.num_task = args.batch_size
        self.num_to_select = 0

        self.model = models.create(args.arch, args)
        # self.model.classifier.apply(weight_init)
        self.model = nn.DataParallel(self.model).cuda()

        #GNN
        self.gnnModel = models.create('gnn', args)
        self.gnnModel = nn.DataParallel(self.gnnModel).cuda()
        # self.classifer = Classifier(self.args).cuda()

        self.meter = meter(args.num_class)
        self.v = v
        # CE for node

        if args.loss == 'focal':
            # self.criterionCE = FocalLoss(self.data.alpha_value).cuda()
            self.criterionCE = FocalLoss().cuda()
        elif args.loss == 'nll':
            self.criterionCE = nn.NLLLoss(reduction='mean').cuda()

        # BCE for edge
        self.criterion = nn.BCELoss(reduction='mean').cuda()
        self.global_step = 0
        self.logger = logger
        self.val_acc = 0
        self.threshold = args.threshold

        ####

        if self.args.discriminator:
            self.discriminator = Discriminator(self.args.in_features)
            self.discriminator = nn.DataParallel(self.discriminator).cuda()

    def get_dataloader(self, dataset, training=False):

        if self.args.visualization:
            data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_workers,
                                     shuffle=training, pin_memory=True, drop_last=True)
            return data_loader

        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_workers,
                                 shuffle=training, pin_memory=True, drop_last=training)
        return data_loader

    def adjust_lr(self, epoch, step_size):
        lr = self.args.lr / (2 ** (epoch // step_size))
        for g in self.optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

        if epoch % step_size == 0:
            print("Epoch {}, current lr {}".format(epoch, lr))

    def label2edge(self, targets):
        '''
        creat initial edge map and edge mask for unlabeled targets
        '''
        batch_size, num_sample = targets.size()
        target_node_mask = torch.eq(targets, self.num_class).type(torch.bool).cuda()
        source_node_mask = ~target_node_mask & ~torch.eq(targets, self.num_class - 1).type(torch.bool)

        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)

        edge = torch.eq(label_i, label_j).float().cuda()
        # unlabeled flag
        target_edge_mask = (torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class)).type(torch.bool).cuda()
        # TO-DO: consider the unk inner connection
        # unk_self_edge_mask = (torch.eq(label_i, self.num_class - 1) & torch.eq(label_j, self.num_class - 1)).type(torch.bool).cuda()
        source_edge_mask = ~target_edge_mask
        # source_edge_mask[unk_self_edge_mask] = ~source_edge_mask[unk_self_edge_mask]
        init_edge = (edge*source_edge_mask.float())# - 1 * unk_self_edge_mask.float()).cuda()

        return init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask


    def transform_shape(self, tensor):

        batch_size, num_class, other_dim = tensor.shape
        tensor = tensor.view(1, batch_size * num_class, other_dim)
        return tensor

    def train(self, step, epochs=70, step_size=55):
        args = self.args

        train_loader = self.get_dataloader(self.data, training=True)

        # initialize model

        # change the learning rate
        if args.arch == 'res':
            if args.dataset == 'visda' or args.dataset == 'office' or args.dataset == 'visda18':
                param_groups = [
                        {'params': self.model.module.CNN.parameters(), 'lr_mult': 0.01},
                        {'params': self.gnnModel.parameters(), 'lr_mult': 0.08},
                    ]
                if self.args.discriminator:
                    param_groups.append({'params': self.discriminator.parameters(), 'lr_mult': 0.1})
            else:
                param_groups = [
                    {'params': self.model.module.CNN.parameters(), 'lr_mult': 0.05},
                    {'params': self.gnnModel.parameters(), 'lr_mult': 0.8},
                ]
                if self.args.discriminator:
                    param_groups.append({'params': self.discriminator.parameters(), 'lr_mult': 0.8})


            args.in_features = 2048

        elif args.arch == 'vgg':
            param_groups = [
                {'params': self.model.module.extractor.parameters(), 'lr_mult': 1},
                {'params': self.gnnModel.parameters(), 'lr_mult': 1},
            ]
            args.in_features = 4096
        #
        self.optimizer = torch.optim.Adam(params=param_groups,
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.model.train()
        self.gnnModel.train()
        self.meter.reset()

        for epoch in range(epochs):
            self.adjust_lr(epoch, step_size)

            with tqdm(total=len(train_loader)) as pbar:
                for i, inputs in enumerate(train_loader):

                    images = Variable(inputs[0], requires_grad=False).cuda()
                    targets = Variable(inputs[1]).cuda()

                    targets_DT = targets[:, args.num_class - 1:].reshape(-1)
                    # only for debugging
                    target_labels = Variable(inputs[2]).cuda()

                    if self.args.discriminator:
                        domain_label = Variable(inputs[3].float()).cuda()

                    targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)
                    target_labels = self.transform_shape(target_labels.unsqueeze(-1)).view(-1)

                    init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)


                    # extract backbone features
                    features = self.model(images)
                    features = self.transform_shape(features)



                    # feed into graph networks
                    edge_logits, node_logits = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge,
                                                             target_mask=target_edge_mask)

                    # compute edge loss
                    full_edge_loss = [self.criterion(edge_logit.masked_select(source_edge_mask), init_edge.masked_select(source_edge_mask)) for edge_logit in edge_logits]
                    norm_node_logits = F.softmax(node_logits[-1], dim=-1)

                    if args.loss == 'nll':
                        source_node_loss = self.criterionCE(torch.log(norm_node_logits[source_node_mask, :] + 1e-5),
                                                            targets.masked_select(source_node_mask))


                    elif args.loss == 'focal':
                        source_node_loss = self.criterionCE(norm_node_logits[source_node_mask, :],
                                                            targets.masked_select(source_node_mask))

                    edge_loss = 0
                    for l in range(args.num_layers - 1):
                        edge_loss += full_edge_loss[l] * 0.5
                    edge_loss += full_edge_loss[-1] * 1
                    loss = 1 * edge_loss + args.node_loss* source_node_loss

                    if self.args.discriminator:
                        unk_label_mask = torch.eq(targets, args.num_class-1).squeeze()
                        domain_pred = self.discriminator(features)
                        temp = domain_pred.view(-1)[~unk_label_mask]
                        domain_loss = self.criterion(temp, domain_label.view(-1)[~unk_label_mask]) #(targets.size(1) / temp.size(0)) *
                        loss = loss + args.adv_coeff * domain_loss

                    node_pred = norm_node_logits[source_node_mask, :].detach().cpu().max(1)[1]
                    node_prec = node_pred.eq(targets.masked_select(source_node_mask).detach().cpu()).double().mean()

                    # Only for debugging
                    if target_node_mask.any():

                        target_pred = norm_node_logits[target_node_mask, :].max(1)[1]

                        # <unlabeled> data mask
                        pseudo_label_mask = ~torch.eq(targets_DT, args.num_class).detach().cpu()

                        # remove unk for calculation
                        unk_label_mask = torch.eq(target_labels[~pseudo_label_mask], args.num_class - 1).detach().cpu()

                        # only predict on <unlabeled> data
                        target_prec = target_pred.eq(target_labels[~pseudo_label_mask]).double().data.cpu()

                        # update prec calculation on <unlabeled> data
                        self.meter.update(target_labels[~pseudo_label_mask].detach().cpu().view(-1).numpy(),
                                          target_prec.numpy())

                        # For pseudo_labeled data, remove unk data for prec calculation
                        pseudo_unk_mask = torch.eq(target_labels[pseudo_label_mask], args.num_class - 1).detach().cpu()
                        pseudo_prec = torch.eq(target_labels[pseudo_label_mask], targets_DT[pseudo_label_mask]).double()
                        if True in pseudo_unk_mask:
                            self.meter.update(target_labels[pseudo_label_mask].detach().cpu().masked_select(~pseudo_unk_mask).view(-1).numpy(),
                                              pseudo_prec.detach().cpu().masked_select(~pseudo_unk_mask).view(-1).numpy())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.logger.global_step += 1
                    # self.logger.log_scalar('train/edge_loss', edge_loss, self.global_step)
                    # self.logger.log_scalar('train/node_loss', source_node_loss, self.global_step)
                    if self.args.discriminator:
                        self.logger.log_scalar('train/domain_loss', domain_loss, self.logger.global_step)
                    self.logger.log_scalar('train/node_prec', node_prec, self.logger.global_step)
                    self.logger.log_scalar('train/edge_loss', edge_loss, self.logger.global_step)
                    # self.logger.log_scalar('train/lr', self.lr, self.logger.global_step)
                    # for k in range(args.num_class - 1):
                    #     self.logger.log_scalar('train/'+args.class_name[k], self.meter.avg[k], self.logger.global_step)
                    self.logger.log_scalar('train/OS_star', self.meter.avg[:-1].mean(), self.logger.global_step)
                    self.logger.log_scalar('train/OS', self.meter.avg.mean(), self.logger.global_step)
                    pbar.update()
                    if i > 150:
                        break
            if (epoch + 1) % args.log_epoch == 0:
                print('---- Start Epoch {} Training --------'.format(epoch))
                for k in range(args.num_class - 1):
                    print('Target {} Precision: {:.3f}'.format(args.class_name[k], self.meter.avg[k]))

                print('Step: {} | {}; Epoch: {}\t'
                      'Training Loss {:.3f}\t'
                      'Training Prec {:.3%}\t'
                      'Target Prec {:.3%}\t'
                      .format(self.logger.global_step, len(train_loader), epoch, loss.data.cpu().numpy(),
                              node_prec.data.cpu().numpy(), self.meter.avg[:-1].mean()))
                self.meter.reset()

        # save model
        states = {'model': self.model.state_dict(),
                  'graph': self.gnnModel.state_dict(),
                  'iteration': self.logger.global_step,
                  'val_acc': node_prec,
                  'optimizer': self.optimizer.state_dict()}
        torch.save(states, osp.join(args.checkpoints_dir, '{}_step_{}.pth.tar'.format(args.experiment, step)))
        self.meter.reset()

    # def trainwognn(self, step, epochs=70, step_size=55):
    #     args = self.args
    #
    #     train_loader = self.get_dataloader(self.data, training=True)
    #
    #     # initialize model
    #
    #     # change the learning rate
    #     if args.arch == 'res':
    #         if args.dataset == 'visda' or args.dataset == 'office' or args.dataset == 'visda18':
    #             param_groups = [
    #                     {'params': self.model.module.CNN.parameters(), 'lr_mult': 0.01},
    #                     {'params': self.classifer.parameters(), 'lr_mult': 0.8},
    #                 ]
    #             if self.args.discriminator:
    #                 param_groups.append({'params': self.discriminator.parameters(), 'lr_mult': 0.1})
    #         args.in_features = 2048
    #
    #     elif args.arch == 'vgg':
    #         param_groups = [
    #             {'params': self.model.module.extractor.parameters(), 'lr_mult': 1},
    #             {'params': self.gnnModel.parameters(), 'lr_mult': 1},
    #         ]
    #         args.in_features = 4096
    #     #
    #     self.optimizer = torch.optim.Adam(params=param_groups,
    #                                       lr=args.lr,
    #                                       weight_decay=args.weight_decay)
    #     self.model.train()
    #     self.classifer.train()
    #     self.meter.reset()
    #
    #     for epoch in range(epochs):
    #         self.adjust_lr(epoch, step_size)
    #
    #         with tqdm(total=len(train_loader)) as pbar:
    #             for i, inputs in enumerate(train_loader):
    #
    #                 images = Variable(inputs[0], requires_grad=False).cuda()
    #                 targets = Variable(inputs[1]).cuda()
    #
    #                 targets_DT = targets[:, args.num_class - 1:].reshape(-1)
    #                 # only for debugging
    #                 target_labels = Variable(inputs[2]).cuda()
    #
    #                 if self.args.discriminator:
    #                     domain_label = Variable(inputs[3].float()).cuda()
    #
    #                 targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)
    #                 target_labels = self.transform_shape(target_labels.unsqueeze(-1)).view(-1)
    #
    #                 init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(
    #                     targets)
    #
    #                 # extract backbone features
    #                 features = self.model(images)
    #                 features = self.transform_shape(features)
    #                 logits = self.classifer(features)
    #                 logits = F.softmax(logits[-1], dim=-1)
    #                 source_node_loss = self.criterionCE(torch.log(logits[source_node_mask.squeeze()] + 1e-5),
    #                                                     targets.masked_select(source_node_mask))
    #
    #
    #                 if self.args.discriminator:
    #                     unk_label_mask = torch.eq(targets, args.num_class-1).squeeze()
    #                     domain_pred = self.discriminator(features)
    #                     temp = domain_pred.view(-1)[~unk_label_mask]
    #                     domain_loss = self.criterion(temp, domain_label.view(-1)[~unk_label_mask])
    #                     loss = source_node_loss + args.adv_coeff * (targets.size(1) / temp.size(0)) * domain_loss
    #
    #                 node_pred = logits[source_node_mask.squeeze()].detach().cpu().max(1)[1]
    #                 node_prec = node_pred.eq(targets.masked_select(source_node_mask).detach().cpu()).double().mean()
    #
    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 self.optimizer.step()
    #
    #                 self.logger.global_step += 1
    #                 # self.logger.log_scalar('train/edge_loss', edge_loss, self.global_step)
    #                 # self.logger.log_scalar('train/node_loss', source_node_loss, self.global_step)
    #                 if self.args.discriminator:
    #                     self.logger.log_scalar('train/domain_loss', domain_loss, self.logger.global_step)
    #                 self.logger.log_scalar('train/node_prec', node_prec, self.logger.global_step)
    #                 # self.logger.log_scalar('train/lr', self.lr, self.logger.global_step)
    #                 # for k in range(args.num_class - 1):
    #                 #     self.logger.log_scalar('train/'+args.class_name[k], self.meter.avg[k], self.logger.global_step)
    #                 self.logger.log_scalar('train/OS_star', self.meter.avg[:-1].mean(), self.logger.global_step)
    #                 self.logger.log_scalar('train/OS', self.meter.avg.mean(), self.logger.global_step)
    #                 pbar.update()
    #                 if i > 300:
    #                     break
    #         if (epoch + 1) % args.log_epoch == 0:
    #             print('---- Start Epoch {} Training --------'.format(epoch))
    #             for k in range(args.num_class - 1):
    #                 print('Target {} Precision: {:.3f}'.format(args.class_name[k], self.meter.avg[k]))
    #
    #             print('Step: {} | {}; Epoch: {}\t'
    #                   'Training Loss {:.3f}\t'
    #                   'Training Prec {:.3%}\t'
    #                   'Target Prec {:.3%}\t'
    #                   .format(self.logger.global_step, len(train_loader), epoch, loss.data.cpu().numpy(),
    #                           node_prec.data.cpu().numpy(), self.meter.avg[:-1].mean()))
    #             self.meter.reset()
    #
    #     # save model
    #     # states = {'model': self.model.state_dict(),
    #     #           'graph': self.gnnModel.state_dict(),
    #     #           'iteration': self.logger.global_step,
    #     #           'val_acc': node_prec,
    #     #           'optimizer': self.optimizer.state_dict()}
    #     # torch.save(states, osp.join(args.checkpoints_dir, '{}_step_{}.pth.tar'.format(args.experiment, step)))
    #     self.meter.reset()



    def estimate_label(self):

        args = self.args
        print('label estimation...')
        if args.dataset == 'visda':
            test_data = Visda_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag, target_ratio=self.step * args.EF / 100)
        elif args.dataset == 'office':
            test_data = Office_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag,
                                       source=args.source_name, target=args.target_name, target_ratio=self.step * args.EF / 100)
        elif args.dataset == 'home':
            test_data = Home_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag, source=args.source_name,
                              target=args.target_name, target_ratio=self.step * args.EF / 100)
        elif args.dataset == 'visda18':
            test_data = Visda18_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag,
                                      target_ratio=self.step * args.EF / 100)

        self.meter.reset()
        # append labels and scores for target samples
        pred_labels = []
        pred_scores = []
        real_labels = []
        target_loader = self.get_dataloader(test_data, training=False)
        self.model.eval()
        self.gnnModel.eval()
        # self.classifer.eval()
        num_correct = 0
        with tqdm(total=len(target_loader)) as pbar:
            for i, (images, targets, target_labels, _, split) in enumerate(target_loader):

                images = Variable(images, requires_grad=False).cuda()
                targets = Variable(targets).cuda()

                # only for debugging
                target_labels = Variable(target_labels).cuda()
                target_labels = self.transform_shape(target_labels.unsqueeze(-1)).view(-1)

                targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)
                init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)

                # extract backbone features
                features = self.model(images)
                features = self.transform_shape(features)
                torch.cuda.empty_cache()
                # feed into graph networks
                edge_logits, node_logits = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge,
                                                         target_mask=target_edge_mask)
                # node_logits = self.classifer(features)
                del features
                norm_node_logits = F.softmax(node_logits[-1], dim=-1)
                # norm_node_logits = F.softmax(node_logits, dim=-1)
                target_score, target_pred = norm_node_logits[target_node_mask, :].max(1)
                # Unk Classification by using Edge

                # edge_pred = edge_logits[-1].detach().cpu()
                # class_logits = {key: [] for key in range(self.num_class - 1)}
                # source_to_target = edge_pred.squeeze(0)[source_node_mask.squeeze(0), :][:, target_node_mask.squeeze(0)]
                # for index, label in enumerate(targets[source_node_mask].detach().cpu().numpy()):
                #     class_logits[label].append(source_to_target[index, :])
                # class_logits = [torch.mean(torch.stack(class_logits[key]), 0) for key in class_logits.keys()]
                # unk_index = torch.max(torch.stack(class_logits), 0)[0] > 0.05

                # relation_to_known = edge_pred[:, :, source_node_mask.squeeze(0)] > 0.0001
                # unk = (~relation_to_known).all(dim=2)

                pred = target_pred.detach().cpu()
                # pred[unk_index] = self.args.num_class - 1
                # only for debugunk_index
                # remove unk for calculation
                # unk_label_mask = torch.eq(target_labels, args.num_class - 1).detach().cpu()
                target_prec = pred.eq(target_labels.detach().cpu()).double()

                self.meter.update(
                    target_labels.detach().cpu().view(-1).data.cpu().numpy(),
                    target_prec.numpy())



                pred_labels.append(target_pred.cpu().detach())
                pred_scores.append(target_score.cpu().detach())
                real_labels.append(target_labels.cpu().detach())

                if i % self.args.log_step == 0:
                    print('Step: {} | {}; \t'
                          'OS Prec {:.3%}\t'
                          .format(i, len(target_loader),
                                  self.meter.avg.mean()))

                pbar.update()


        pred_labels = torch.cat(pred_labels)
        pred_scores = torch.cat(pred_scores)
        real_labels = torch.cat(real_labels)



        self.model.train()
        self.gnnModel.train()
        # self.classifer.train()
        # self.num_to_select = int(self.meter.count.sum() * (self.step + 1) * self.args.EF / 100)
        # TO-DO
        self.num_to_select = int(len(target_loader) * self.args.batch_size * (self.args.num_class - 1) * self.args.EF / 100)
        return pred_labels.data.cpu().numpy(), pred_scores.data.cpu().numpy(), real_labels.data.cpu().numpy()

    def select_top_data(self, pred_score):
        # remark samples if needs pseudo labels based on classification confidence
        if self.v is None:
            self.v = np.zeros(len(pred_score))
        unselected_idx = np.where(self.v == 0)[0]
        if len(unselected_idx) < self.num_to_select:
            self.num_to_select = len(unselected_idx)
        index = np.argsort(-pred_score[unselected_idx])
        index_orig = unselected_idx[index]
        num_pos = int(self.num_to_select * self.threshold)
        num_neg = self.num_to_select - num_pos
        for i in range(num_pos):
            self.v[index_orig[i]] = 1
        for i in range(num_neg):
            self.v[index_orig[-i]] = -1
        return self.v

    def generate_new_train_data(self, sel_idx, pred_y, real_label):
        # create the new dataset merged with pseudo labels
        assert len(sel_idx) == len(pred_y)
        new_label_flag = []
        pos_correct, pos_total, neg_correct, neg_total = 0, 0, 0, 0
        for i, flag in enumerate(sel_idx):
            if i >= len(real_label):
                break
            if flag > 0:
                new_label_flag.append(pred_y[i])
                pos_total += 1
                if real_label[i] == pred_y[i]:
                    pos_correct += 1
            elif flag < 0:
                # assign the <unk> pseudo label
                new_label_flag.append(self.args.num_class - 1)
                pred_y[i] = self.args.num_class - 1
                neg_total += 1
                if real_label[i] == self.args.num_class - 1:
                    neg_correct += 1
            else:
                new_label_flag.append(self.args.num_class)


        self.meter.reset()
        self.meter.update(real_label, (pred_y == real_label).astype(int))

        for k in range(self.args.num_class):
            print('Target {} Precision: {:.3f}'.format(self.args.class_name[k], self.meter.avg[k]))

        for k in range(self.num_class):
            self.logger.log_scalar('test/' + self.args.class_name[k], self.meter.avg[k], self.step)
        self.logger.log_scalar('test/ALL', self.meter.sum.sum() / self.meter.count.sum(), self.step)
        self.logger.log_scalar('test/OS_star', self.meter.avg[:-1].mean(), self.step)
        self.logger.log_scalar('test/OS', self.meter.avg.mean(), self.step)

        print('Node predictions: OS accuracy = {:0.4f}, OS* accuracy = {:0.4f}'.format(self.meter.avg.mean(), self.meter.avg[:-1].mean()))

        correct = pos_correct + neg_correct
        total = pos_total + neg_total
        acc = correct / total
        pos_acc = pos_correct / pos_total
        neg_acc = neg_correct / neg_total
        new_label_flag = torch.tensor(new_label_flag)

        # update source data
        if self.args.dataset == 'visda':
            new_data = Visda_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                     target_ratio=(self.step + 1) * self.args.EF / 100)

        elif self.args.dataset == 'office':
            new_data = Office_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                       source=self.args.source_name, target=self.args.target_name,
                                      target_ratio=(self.step + 1) * self.args.EF / 100)

        elif self.args.dataset == 'home':
            new_data = Home_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                    source=self.args.source_name, target=self.args.target_name,
                                    target_ratio=(self.step + 1) * self.args.EF / 100)
        elif self.args.dataset == 'visda18':
            new_data = Visda18_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                     target_ratio=(self.step + 1) * self.args.EF / 100)

        print('selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}'.format(correct, total, acc))
        print('positive data: {} of {} is correct, accuracy: {:0.4f}'.format(pos_correct, pos_total, pos_acc))
        print('negative data: {} of {} is correct, accuracy: {:0.4f}'.format(neg_correct, neg_total, neg_acc))
        return new_label_flag, new_data

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes, dtype=torch.long)[class_idx]

    def load_model_weight(self, path):
        print('loading weight')
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.gnnModel.load_state_dict(state['graph'])

    def label2edge_gt(self, targets):
        '''
        creat initial edge map and edge mask for unlabeled targets
        '''
        batch_size, num_sample = targets.size()
        target_node_mask = torch.eq(targets, self.num_class).type(torch.bool).cuda()
        source_node_mask = ~target_node_mask & ~torch.eq(targets, self.num_class - 1).type(torch.bool)

        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)

        edge = torch.eq(label_i, label_j).float().cuda()
        target_edge_mask = (torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class)).type(
            torch.bool).cuda()
        source_edge_mask = ~target_edge_mask
        # unlabeled flag

        return (edge*source_edge_mask.float())

    def extract_feature(self):
        print('Feature extracting...')
        self.meter.reset()
        # append labels and scores for target samples
        vgg_features_target = []
        node_features_target = []
        labels = []
        overall_split = []
        target_loader = self.get_dataloader(self.data, training=False)
        self.model.eval()
        self.gnnModel.eval()
        num_correct = 0
        skip_flag = self.args.visualization
        with tqdm(total=len(target_loader)) as pbar:
            for i, (images, targets, target_labels, _, split) in enumerate(target_loader):

                # for debugging
                # if i > 100:
                #     break
                images = Variable(images, requires_grad=False).cuda()
                targets = Variable(targets).cuda()

                # only for debugging
                # target_labels = Variable(target_labels).cuda()

                targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)
                target_labels = self.transform_shape(target_labels.unsqueeze(-1)).squeeze(-1).cuda()
                init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)
                # gt_edge = self.label2edge_gt(target_labels)
                # extract backbone features
                features = self.model(images)
                features = self.transform_shape(features)

                # feed into graph networks
                edge_logits, node_feat = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge,
                                                         target_mask=target_edge_mask)
                vgg_features_target.append(features.data.cpu())
                #####heat map only
                # temp = np.array(edge_logits[0].data.cpu()) * 4
                # ax = sns.heatmap(temp.squeeze(), vmax=1)#
                # cbar = ax.collections[0].colorbar
                # # here set the labelsize by 20
                # cbar.ax.tick_params(labelsize=17)
                # plt.savefig('heat/' + str(i) + '.png')
                # plt.close()
                ###########
                node_features_target.append(node_feat[-1].data.cpu())
                labels.append(target_labels.data.cpu())
                overall_split.append(split)
                if skip_flag and i > 50:
                    break

                pbar.update()

        return vgg_features_target, node_features_target, labels, overall_split









