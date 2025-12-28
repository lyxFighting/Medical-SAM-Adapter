import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # ---------- load image & mask ----------
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # resize mask to image size (for consistency)
        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        # ---------- transform ----------
        if self.transform:
            img = self.transform(img)

        # mask → tensor (1, H, W), value in [0,1]
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

        # =====================================================
        # 🔥 Weak GT mask → SAM mask prompt
        # =====================================================
        import torch.nn.functional as F

        # 1. resize to 256×256 (SAM requirement)
        mask_prompt = F.interpolate(
            mask.unsqueeze(0),        # (1, 1, H, W)
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        # 2. blur / coarsen
        mask_prompt = F.avg_pool2d(
            mask_prompt,
            kernel_size=8,
            stride=8
        )
        mask_prompt = F.interpolate(
            mask_prompt,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        # 3. threshold → coarse mask
        mask_prompt = (mask_prompt > 0.3).float()
        mask_prompt = mask_prompt.squeeze(0)   # (1, 256, 256)

        # ---------- meta ----------
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': img,                   # (3, H, W)
            'mask_prompt': mask_prompt,     # (1, 256, 256)
            'image_meta_dict': image_meta_dict,
        }
