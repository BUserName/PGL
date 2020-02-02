'''
Todo: resume / Tensorboar
Modified Data: 31/10/2019
Author: Yadan Luo, Zijian Wang
'''

from __future__ import print_function, absolute_import
import os
import argparse
import random
import numpy as np
# torch-related packages
import torch
import matplotlib.pyplot as plt
from utils.visualization import visualize_TSNE

# Modified here
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import tensorboardX

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# data
from data_loader import Visda_Dataset, Office_Dataset, Home_Dataset, Visda18_Dataset
from model_trainer import ModelTrainer

# Modified here
from utils.logger import Logger

def main(args):
    # Modified here
    total_step = 100//args.EF

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # prepare checkpoints and log folders
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # initialize dataset
    if args.dataset == 'visda':
        args.data_dir = os.path.join(args.data_dir, 'visda')
        data = Visda_Dataset(root=args.data_dir, partition='train', label_flag=None)

    elif args.dataset == 'office':
        args.data_dir = os.path.join(args.data_dir, 'Office')
        data = Office_Dataset(root=args.data_dir, partition='train', label_flag=None, source=args.source_name,
                              target=args.target_name)

    elif args.dataset == 'home':
        args.data_dir = os.path.join(args.data_dir, 'OfficeHome')
        data = Home_Dataset(root=args.data_dir, partition='train', label_flag=None, source=args.source_name,
                              target=args.target_name)
    elif args.dataset == 'visda18':
        args.data_dir = os.path.join(args.data_dir, 'visda18')
        data = Visda18_Dataset(root=args.data_dir, partition='train', label_flag=None)
    else:
        print('Unknown dataset!')

    args.class_name = data.class_name
    args.num_class = data.num_class
    args.alpha = data.alpha
    # setting experiment name
    label_flag = None
    selected_idx = None
    args.experiment = set_exp_name(args)

    # Modified here
    logger = Logger(args)

    if not args.visualization:

        for step in range(total_step):

            print("This is {}-th step with EF={}%".format(step, args.EF))

            # Modified here
            trainer = ModelTrainer(args=args, data=data, step=step, label_flag=label_flag, v=selected_idx, logger=logger)

            # train the model
            args.log_epoch = 4 + step//2
            # 10 + (step // 2) * args.log_epoch
            trainer.train(step, epochs= 4 + (step) * 2, step_size=args.log_epoch)
            #trainer.trainwognn(step, epochs= 2 + (step) * 2 , step_size=args.log_epoch)#
            # psedo_label
            pred_y, pred_score, pred_acc = trainer.estimate_label()

            # select data from target to source
            selected_idx = trainer.select_top_data(pred_score)

            # add new data
            label_flag, data = trainer.generate_new_train_data(selected_idx, pred_y, pred_acc)
    else:
        # load trained weights
        trainer = ModelTrainer(args=args, data=data)
        trainer.load_model_weight(args.checkpoint_path)
        vgg_feat, node_feat, target_labels, split = trainer.extract_feature()
        visualize_TSNE(node_feat, target_labels, args.num_class, args, split)

        plt.savefig('./node_tsne.png', dpi=300)



def set_exp_name(args):
    exp_name = 'D-{}'.format(args.dataset)
    if args.dataset == 'office' or args.dataset == 'home':
        exp_name += '_src-{}_tar-{}'.format(args.source_name, args.target_name)
    exp_name += '_A-{}'.format(args.arch)
    exp_name += '_L-{}'.format(args.num_layers)
    exp_name += '_E-{}_B-{}'.format(args.EF, args.batch_size)
    return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exploit Domain Adaptation')
    # set up dataset & backbone embedding
    dataset = 'visda18'
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('-a', '--arch', type=str, default='res')
    parser.add_argument('--root_path', type=str, default='./utils/', metavar='B',
                        help='root dir')

    # set up path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'data/'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'logs'))
    parser.add_argument('--checkpoints_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'checkpoints'))

    # verbose setting
    parser.add_argument('--log_step', type=int, default=30)
    parser.add_argument('--log_epoch', type=int, default=3)

    if dataset == 'office':
        parser.add_argument('--source_name', type=str, default='D')
        parser.add_argument('--target_name', type=str, default='W')

    elif dataset == 'home':
        parser.add_argument('--source_name', type=str, default='R')
        parser.add_argument('--target_name', type=str, default='A')
    else:
        print("Set a log step first !")
    parser.add_argument('--eval_log_step', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=1500)

    # whether resume

    parser.add_argument('--resume', type=str, default=None)

    # hyper-parameters

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('-b', '--batch_size', type=int, default=4)
    # parser.add_argument('--num_task', type=int, default=2)

    # parser.add_argument('--epoch', type=int, default=8)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('--threshold', type=float, default=0.6)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--EF', type=int, default=100)
    parser.add_argument('--loss', type=str, default='focal')


    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)


    #GNN parameters
    # res 2048 vgg 4096
    # if dataset == 'visda':
    #     parser.add_argument('--in_features', type=int, default=2048)
    # elif dataset == 'office':
    #     parser.add_argument('--in_features', type=int, default=2048)
    # elif dataset == 'home':
    #     parser.add_argument('--in_features', type=int, default=2048)

    parser.add_argument('--in_features', type=int, default=2048)
    if dataset == 'home':
        parser.add_argument('--node_features', type=int, default=512)
        parser.add_argument('--edge_features', type=int, default=512)
    else:
        parser.add_argument('--node_features', type=int, default=1024)
        parser.add_argument('--edge_features', type=int, default=1024)

    parser.add_argument('--num_layers', type=int, default=1)

    #tsne
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='/home/zijian/Desktop/Open_DA_git/checkpoints/D-visda18_A-res_L-1_E-20_B-4_step_1.pth.tar')

    #Discrminator
    parser.add_argument('--discriminator', type=bool, default=True)
    parser.add_argument('--adv_coeff', type=float, default=0.4)


    #GNN hyper-parameters
    parser.add_argument('--node_loss', type=float, default=0.3)
    main(parser.parse_args())

