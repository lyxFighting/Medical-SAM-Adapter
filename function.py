
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
#from models.discriminatorlayer import discriminator
from conf import settings
from utils import *

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(
    args,
    net: nn.Module,
    optimizer,
    train_loader,
    epoch,
    writer,
    schedulers=None,
    vis=50
):
    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # loss
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    epoch_loss = 0
    ind = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            ind += 1

            # ====================================================
            # 1. Load data (NEW FORMAT)
            # ====================================================
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            mask_prompt = pack['mask_prompt'].to(dtype=torch.float32, device=GPUdevice)#Dataset 里是 (1,256,256)，但 DataLoader 自动加了 batch 维
            name = pack['image_meta_dict']['filename_or_obj']

            # ====================================================
            # 2. Freeze / unfreeze parameters
            # ====================================================
            if args.mod == 'sam_adpt':
                for n, p in net.image_encoder.named_parameters():
                    p.requires_grad = ("Adapter" in n)
            elif args.mod in ['sam_lora', 'sam_adalora']:
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    rankallocator = lora.RankAllocator(
                        net.mage_encoder,
                        lora_r=4,
                        target_rank=8,
                        init_warmup=500,
                        final_warmup=1500,
                        mask_interval=10,
                        total_step=3000,
                        beta1=0.85,
                        beta2=0.85,
                    )
            else:
                for _, p in net.image_encoder.named_parameters():
                    p.requires_grad = True
      
            # ====================================================
            # 3. Image encoder
            # ====================================================
            origin_imgs = imgs.clone()
            imgs = net.preprocess(imgs)
            image_embeddings = net.image_encoder(imgs)

            # ====================================================
            # 4. Prompt encoder (🔥 mask prompt only)
            # ====================================================
            with torch.no_grad():
                if args.net in ['sam', 'mobile_sam']:
                    se, de = net.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=mask_prompt,
                    )
                elif args.net == 'efficient_sam':
                    se = net.prompt_encoder(
                        masks=mask_prompt,
                    )

            # ====================================================
            # 5. Mask decoder
            # ====================================================
            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                )
            elif args.net == 'efficient_sam':
                se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                pred, _ = net.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    multimask_output=False,
                )

            # ====================================================
            # 6. Resize + loss
            # ====================================================
            pred = F.interpolate(
                pred,
                size=(args.out_size, args.out_size),
                mode="bilinear",
                align_corners=False
            )

            loss = lossfunc(pred, masks)
            loss2=lossfunc(mask_prompt,masks)
            print('mask prompt和gt的损失值：', loss2.item())
            print('pre和gt的损失值：', loss.item())
            epoch_loss += loss.item()

            # ====================================================
            # 7. Backprop
            # ====================================================
            if args.mod == 'sam_adalora':
                (loss + lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            # ====================================================
            # 8. Visualization
            # ====================================================
            if vis and ind % vis == 0:
                vis_image(
                    origin_imgs / 255,
                    pred,
                    masks,
                    os.path.join(
                        args.path_helper['sample_path'],
                        f"Train_{name}_epoch_{epoch}.jpg"
                    ),
                    reverse=False
                )

            pbar.set_postfix({'loss': loss.item()})#在进度条尾部动态显示当前迭代的 loss（损失值），让训练过程更直观
            pbar.update()

    return epoch_loss / len(train_loader)


def validation_sam(
    args,
    val_loader,
    epoch,
    net: nn.Module,
    clean_dir=True
):
    net.eval()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    dataset_size = len(val_loader.dataset)

    # loss
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    tot_loss = 0
    mix_res = (0,) * args.multimask_output * 2
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

    with torch.no_grad():
        with tqdm(
            total=len(val_loader),
            desc='Validation',
            unit='batch',
            leave=False
        ) as pbar:
            for ind, pack in enumerate(val_loader):
                # ====================================================
                # 1. Load data (NEW FORMAT)
                # ====================================================
                imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
                masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
                mask_prompt = pack['mask_prompt'].to(dtype=torch.float32, device=GPUdevice)
                name = pack['image_meta_dict']['filename_or_obj']

                cur_bsz = imgs.shape[0]


                # ====================================================
                # 2. Forward
                # ====================================================
                origin_imgs = imgs.clone()
                imgs = net.preprocess(imgs)
                image_embeddings = net.image_encoder(imgs)

                # -------- prompt encoder (mask only) --------
                if args.net in ['sam', 'mobile_sam']:
                    se, de = net.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=mask_prompt,
                    )
                elif args.net == "efficient_sam":
                    se = net.prompt_encoder(
                        masks=mask_prompt,
                    )

                # -------- mask decoder --------
                if args.net == 'sam':
                    pred, _ = net.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=(args.multimask_output > 1),
                    )
                elif args.net == 'mobile_sam':
                    pred, _ = net.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )
                elif args.net == "efficient_sam":
                    se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                    pred, _ = net.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        multimask_output=False,
                    )

                # ====================================================
                # 3. Resize + loss
                # ====================================================
                pred = F.interpolate(
                    pred,
                    size=(args.out_size, args.out_size),
                    mode="bilinear",
                    align_corners=False
                )

                loss = lossfunc(pred, masks)
                # loss2=lossfunc(mask_prompt,masks)
                # print('mask prompt和gt的损失值：', loss2.item())
                # print('pre和gt的损失值：', loss.item())
                tot_loss += loss.item() * cur_bsz

                # ====================================================
                # 4. Metrics
                # ====================================================
                temp = eval_seg(pred, masks, threshold)
                temp = tuple([v * cur_bsz for v in temp])
                mix_res = tuple(sum(x) for x in zip(mix_res, temp))

                # ====================================================
                # 5. Visualization
                # ====================================================
                if args.vis and ind % args.vis == 0:
                    namecat = "Val_"
                    for na in name[:2]:
                        img_name = na.split('/')[-1].split('.')[0]
                        namecat += img_name + '+'

                    vis_image(
                        origin_imgs / 255,
                        pred,
                        masks,
                        os.path.join(
                            args.path_helper['sample_path'],
                            f"{namecat}epoch_{epoch}.jpg"
                        ),
                        reverse=False
                    )

                pbar.update()

    return (
        tot_loss / dataset_size,
        tuple(v / dataset_size for v in mix_res)
    )

