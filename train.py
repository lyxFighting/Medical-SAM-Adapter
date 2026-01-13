import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
import function
from conf import settings
from dataset import *
from utils import *

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = "/home/liuyuxiu/.local/share/fonts/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]
plt.rcParams['axes.unicode_minus'] = False



def main():
    args = cfg.parse_args()
    seed = args.seed
    set_seed(seed)

    data_path='/home/liuyuxiu/models/Medical-SAM-Adapter/data'
    GPUdevice = torch.device('cuda', args.gpu_device)
    # data_list = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    data_list = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))],
                       key=lambda x: len([f for f in os.listdir(os.path.join(data_path, x, 'images')) 
                                          if os.path.isfile(os.path.join(data_path, x, 'images', f))]) 
                       if os.path.exists(os.path.join(data_path, x, 'images')) else 0)
    shared_prefix = create_timestamp_dir('/home/liuyuxiu/models/Medical-SAM-Adapter/logs_alldata')
    data_list=['Fascicle-FALLMUDRyan']
    for dataset_name in data_list:
        args.dataset = dataset_name
        args.data_path = os.path.join(data_path, dataset_name)
        args.path_helper = set_log_dir(shared_prefix, dataset_name)
        logger = create_logger(args.path_helper['log_path'],name=dataset_name)
        logger.info(args)
        print(f'================== Start training on {dataset_name} ==================')

        net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
        if args.pretrain:
            weights = torch.load(args.pretrain)
            net.load_state_dict(weights,strict=False)

        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

        nice_train_loader, nice_test_loader = get_dataloader(args)

        '''checkpoint path and tensorboard'''
        # iter_per_epoch = len(Glaucoma_training_loader)
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
        #use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.net, settings.TIME_NOW))

        '''begain training'''
        best_acc = 0.0
        best_tol = 1e4
        best_dice = 0.0
        epoch_list=[]
        loss_list=[]

        for epoch in range(settings.EPOCH):
            if epoch < 0:
                tol, (eiou, edice,hd95) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, HD95: {hd95}|| @ epoch {epoch}.')

            net.train()
            time_start = time.time()
            loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch,vis = args.vis)
            logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
            epoch_list.append(epoch)
            loss_list.append(loss)
            time_end = time.time()
            print('time_for_training ', time_end - time_start)

            net.eval()
            if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
                tol, (eiou, edice,hd95) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, HD95: {hd95}|| @ epoch {epoch}.')
                if args.distributed != 'none':
                    sd = net.module.state_dict()
                else:
                    sd = net.state_dict()

                if edice > best_dice:
                    best_tol = tol
                    is_best = True
                    best_dice = edice

                    save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.net,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                    'best_tol': best_dice,
                    'path_helper': args.path_helper,
                }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
                else:
                    is_best = False
        plt.figure(figsize=(12, 6))
        plt.scatter(epoch_list, loss_list, color='red', label='预测分割和真实分割的loss', alpha=0.7, s=50)
        plt.plot(epoch_list, loss_list, color='red', linestyle='--', alpha=0.3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{args.path_helper['log_path']}/training_loss.png", dpi=300, bbox_inches='tight')
        writer.close()


if __name__ == '__main__':
    main()
