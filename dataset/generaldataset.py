import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import random_box, random_click

class dataset(Dataset):
    def __init__(self, args, data_path, img_ids, transform=None, transform_msk=None, prompt='click', img_ext='.jpg', mask_ext='.png'):
        """
        args: 包含 image_size
        data_path: 数据根目录，下面有 images 和 masks 文件夹
        img_ids: 图片 ID 列表（不带扩展名）
        transform: 图像增强 transform
        transform_msk: mask 增强 transform
        prompt: 是否使用 click 提示
        img_ext: 图片文件扩展名, 如 '.jpg'
        mask_ext: mask 文件扩展名, 如 '.png'
        """
        self.data_path = data_path
        self.img_ids = img_ids
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt = prompt
        self.img_ext = img_ext
        self.mask_ext = mask_ext

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        point_label = 1  # 前景点 label，固定为 1

        # 图片路径
        img_id = self.img_ids[index]
        img_path = os.path.join(self.data_path, 'images', img_id + self.img_ext)
        msk_path = os.path.join(self.data_path, 'masks', img_id + self.mask_ext)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # resize mask
        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        point_label, pt = random_click(np.array(mask) / 255, point_label)
        state = torch.get_rng_state()
        img = self.transform(img)
        torch.set_rng_state(state)
        mask = self.transform_msk(mask).long()

        image_meta_dict = {'filename_or_obj': img_id}

        img = img.float()
        pt = torch.as_tensor(pt, dtype=torch.float32)
        point_label = torch.as_tensor(point_label, dtype=torch.long)
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }
