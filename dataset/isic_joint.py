import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click

import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode



class ISIC2016(Dataset):
    def __init__(self, args, data_path , joint_transform=None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        pt = None

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.joint_transform:
            img, mask, low_mask = self.joint_transform(img, mask)

        if self.prompt == 'click':
            # mask已经是tensor，形状为(1, H, W)或(H, W)
            if len(mask.shape) == 3:
                mask_np = mask[0].numpy()
            else:
                mask_np = mask.numpy()
            point_label, pt = random_click(mask_np, 1)

        # 处理文件名
        filename = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': filename}

        return {
            'image': img,
            'label': low_mask,
            # 'low_mask': low_mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }

    


class ISICJointTransform2D:
    def __init__(
        self,
        img_size=1024,
        low_img_size=256,
        ori_size=1024,
        crop=None,
        p_flip=0.0,
        p_rota=0.0,
        p_scale=0.0,
        p_gaussn=0.0,
        p_contr=0.0,
        p_gama=0.0,
        color_jitter_params=(0.1, 0.1, 0.1, 0.1),
        p_random_affine=0.0,
    ):
        self.img_size = img_size
        self.low_img_size = low_img_size
        self.ori_size = ori_size
        self.crop = crop

        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_contr = p_contr
        self.p_gama = p_gama
        self.p_random_affine = p_random_affine

        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        else:
            self.color_tf = None

    def __call__(self, image, mask):
        """
        image: PIL RGB
        mask : PIL L (0 / 255)
        """

        # ---------- gamma (image only) ----------
        if np.random.rand() < self.p_gama:
            img = np.array(image).astype(np.float32)
            g = np.random.uniform(1.0, 2.5)
            img = np.power(img / 255.0, 1.0 / g) * 255.0
            image = Image.fromarray(img.astype(np.uint8))

        # ---------- random crop ----------
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

        # ---------- flip ----------
        if np.random.rand() < self.p_flip:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # ---------- rotation ----------
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        # ---------- scale ----------
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1.0, 1.3)
            new_h = int(self.img_size * scale)
            new_w = int(self.img_size * scale)

            image = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR)
            mask = F.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)

            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

        # ---------- gaussian noise (image only) ----------
        if np.random.rand() < self.p_gaussn:
            img = np.array(image)
            noise = np.random.normal(0, 10, img.shape)
            img = np.clip(img + noise, 0, 255)
            image = Image.fromarray(img.astype(np.uint8))

        # ---------- contrast ----------
        if np.random.rand() < self.p_contr:
            image = T.ColorJitter(contrast=(0.8, 2.0))(image)

        # ---------- color jitter ----------
        if self.color_tf:
            image = self.color_tf(image)

        # ---------- random affine ----------
        if np.random.rand() < self.p_random_affine:
            params = T.RandomAffine(180).get_params(
                (-90, 90), (1, 1), (2, 2), (-45, 45), self.crop
            )
            image = F.affine(image, *params)
            mask = F.affine(mask, *params, interpolation=InterpolationMode.NEAREST)

        # ---------- resize to final ----------
        image = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR)
        mask = F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)

        # ---------- to tensor ----------
        image = F.to_tensor(image) * 255.0
        mask = (F.to_tensor(mask) > 0).float()
        low_mask = (F.to_tensor(low_mask) > 0).float()

        return image, mask, low_mask
