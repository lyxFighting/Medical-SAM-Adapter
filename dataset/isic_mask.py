import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click
import torch.nn.functional as F
from utils import *

class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'mask', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

        self.GPUdevice = torch.device('cuda', args.gpu_device)

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
        point_label = 1
        point_label, pt = random_click(np.array(mask) / 255, point_label)
        # ---------- transform ----------
        if self.transform:
            img = self.transform(img)
        if self.transform_msk:
            mask = self.transform_msk(mask).int()#soft类型为什么int后不是全0？

        # =====================================================
        # 🔥 Weak GT mask → SAM mask prompt
        # =====================================================
        # if self.prompt=='mask':
        #     mask_prompt=weakmask(mask)

        # ---------- meta ----------
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': img, # (3, H, W)
            'label': mask,
            'p_label':point_label,# 点提示的标签（前景点=1，背景点=0）
            'pt':pt,# 形状: [N, 2]  # N个点，(x, y)坐标                  
            # 'mask_prompt': mask_prompt,     # (1, 256, 256)
            'image_meta_dict': image_meta_dict,
        }
