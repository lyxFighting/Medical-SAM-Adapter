
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
    vis=50,
    logger=None
):
    hard = 0
    epoch_loss = 0
    ind = 0
    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # loss
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G


    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            ind += 1

            # ====================================================
            # 1. Load data (NEW FORMAT)
            # ====================================================
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)#(2,3,1024,1024)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)#(2,1,256,256)
            mask_prompts = net.swinunet(imgs).to(dtype=torch.float32, device=GPUdevice)
            # resize_transform = transforms.Resize((224, 224))
            # imgs_resized = torch.stack([resize_transform(img) for img in imgs])#(2,3,224,224)
            # mask_prompts = net.swinunet(imgs_resized).to(dtype=torch.float32, device=GPUdevice)#(2,1,224,224)
            # resize_transform2 = transforms.Resize((256, 256))
            # mask_prompts  = torch.stack([resize_transform2(mask_prompt) for mask_prompt in mask_prompts])#(2,1,256,256)
            pt = pack['pt']
            point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']
            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)

                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt[..., [1, 0]]

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if(len(point_labels.shape)==1): # only one point prompt
                    coords_torch, labels_torch, showp = coords_torch.unsqueeze(1), labels_torch.unsqueeze(1), showp.unsqueeze(1)
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            # ====================================================
            # 2. Freeze / unfreeze parameters
            # ====================================================
            if args.mod == 'sam_adpt':
                for n, value in net.sam.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
            elif args.mod in ['sam_lora', 'sam_adalora']:
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.sam.image_encoder)
                if args.mod == 'sam_adalora':
                    rankallocator = lora.RankAllocator(
                        net.sam.image_encoder,
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
                for _, p in net.sam.image_encoder.named_parameters():
                    p.requires_grad = True
      
            # ====================================================
            # 3. Image encoder
            # ====================================================
            origin_imgs = imgs.clone()#(2,3,1024,1024)
            imgs = net.sam.preprocess(imgs)
            image_embeddings = net.sam.image_encoder(imgs)#(2,256,64,64)

            # ====================================================
            # 4. Prompt encoder (🔥 mask prompt only)
            # ====================================================
            with torch.no_grad():
                if args.net in ['sam', 'mobile_sam']:
                    se, de = net.sam.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=mask_prompts ,#(2,1,256,256)
                    )#(2,0,256),(2,256,64,64)
                elif args.net == 'efficient_sam':
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )


            # ====================================================
            # 5. Mask decoder
            # ====================================================
            if args.net == 'sam':
                pred, _ = net.sam.mask_decoder(
                    image_embeddings=image_embeddings,#(2,256,64,64)
                    image_pe=net.sam.prompt_encoder.get_dense_pe(),#(1,256,64,64)
                    sparse_prompt_embeddings=se,#(2,0,256)
                    dense_prompt_embeddings=de,#(2,256,64,64)
                    multimask_output=(args.multimask_output > 1),
                )#(2,1,256,256)
            elif args.net == 'mobile_sam':
                pred, _ = net.sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=net.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                )
            elif args.net == 'efficient_sam':
                se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                pred, _ = net.sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=net.sam.prompt_encoder.get_dense_pe(),
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
 
            loss1 = lossfunc(pred, masks)
            loss2 = lossfunc(mask_prompts ,masks)
            logger.info(f'mask prompt和gt的损失值：{loss2.item()}, pre和gt的损失值：{loss1.item()}')
            loss= 0.1*loss1+0.9*loss2
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
                    origin_imgs / 255,#（2,3,1024,1024）
                    mask_prompts ,#(2,1,256,256)logits
                    pred,#(2,1,256,256)logits
                    masks,#(2,1,256,256)soft mask
                    os.path.join(
                        args.path_helper['sample_path'],
                        f"Train_{name}_epoch_{epoch}.jpg"
                    ),
                    reverse=False,
                    points=showp
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

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    dataset_size = len(val_loader.dataset)
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot_loss = 0.0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    # loss
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            # ====================================================
            # 1. Load data (NEW FORMAT)
            # ====================================================
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)#(2,3,1024,1024)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)#(2,1,256,256)

            cur_bsz = imgsw.shape[0]
            mask_prompts = net.swinunet(imgsw).to(dtype=torch.float32, device=GPUdevice)
            # resize_transform = transforms.Resize((224, 224))
            # imgs_resized = torch.stack([resize_transform(img) for img in imgs])#(2,3,224,224)
            # mask_prompts = net.swinunet(imgs_resized).to(dtype=torch.float32, device=GPUdevice)#(2,1,224,224)
            # resize_transform2 = transforms.Resize((256, 256))
            # mask_prompts  = torch.stack([resize_transform2(mask_prompt) for mask_prompt in mask_prompts])#(2,1,256,256)
            ptw = pack['pt']
            point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw
                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt[..., [1, 0]]

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if(len(point_labels.shape)==1): # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch.unsqueeze(1), labels_torch.unsqueeze(1), showp.unsqueeze(1)
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                # ====================================================
                # 2. Forward
                # ====================================================
                with torch.no_grad():
                    origin_imgs = imgs.clone()
                    imgs = net.sam.preprocess(imgs)
                    image_embeddings = net.sam.image_encoder(imgs)

                    # -------- prompt encoder (mask only) --------
                    if args.net in ['sam', 'mobile_sam']:
                        se, de = net.sam.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=mask_prompts,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    # -------- mask decoder --------
                    if args.net == 'sam':
                        pred, _ = net.sam.mask_decoder(
                            image_embeddings=image_embeddings,
                            image_pe=net.sam.prompt_encoder.get_dense_pe(),#get_dense_pe() 为 SAM 提供全图密集位置编码，弥补 Transformer 置换不变性的缺陷，赋予模型空间位置感知能力；
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=(args.multimask_output > 1),
                        )
                    elif args.net == 'mobile_sam':
                        pred, _ = net.sam.mask_decoder(
                            image_embeddings=image_embeddings,
                            image_pe=net.sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                        pred, _ = net.sam.mask_decoder(
                            image_embeddings=image_embeddings,
                            image_pe=net.sam.prompt_encoder.get_dense_pe(),
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
                # 5. Visualization
                # ====================================================
                if ind % args.vis == 0:
                    namecat = "Test"
                    for na in name[:2]:
                        img_name = na.split('/')[-1].split('.')[0]
                        namecat += img_name + '+'

                    vis_image(origin_imgs/255,mask_prompts,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
                temp = eval_seg(pred, masks, threshold)
                temp = tuple([v * cur_bsz for v in temp])
                mix_res = tuple(sum(x) for x in zip(mix_res, temp))

            pbar.update()
    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)
    return (
        tot_loss / dataset_size,
        tuple(v / dataset_size for v in mix_res)
    )

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )