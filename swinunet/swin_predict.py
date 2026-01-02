import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

# ---------- 工具函数 ----------
def load_image_swin(image_path, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0), np.array(img)
# ---------- 主流程 ----------
def swinmask(args,imgs):
    # ---------- Swin-Unet ----------
    config = get_config(args)
    model = ViT_seg(config,
                    img_size=args.img_size,
                    num_classes=args.num_classes).cuda()
    # model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # 方法A：检查是否是检查点格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载检查点，epoch: {checkpoint.get('epoch', 'N/A')}")
    else:
        # 方法B：如果是纯权重文件
        model.load_state_dict(checkpoint)
    model.eval()

    img_tensor, origin_img = load_image_swin(args.input_image, args.img_size)
    img_tensor = img_tensor.cuda()

    with torch.no_grad():
        swin_pred = model(img_tensor)