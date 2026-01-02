import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# ===== 路径处理 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

# SAM
from segment_anything import sam_model_registry, SamPredictor
import argparse


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


def swin_mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2])


def post_process_mask(mask):
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        default='/home/liuyuxiu/models/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml')
    parser.add_argument('--checkpoint',
                        default='/home/liuyuxiu/models/Swin-Unet/checkpoints/isic/best_model.pth')
    parser.add_argument('--sam_ckpt',
                        default='/home/liuyuxiu/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth')

    parser.add_argument('--input_image',
                        default='/home/liuyuxiu/models/Medical-SAM-Adapter/data/ISIC/ISBI2016_ISIC_Part1_Test_Data/ISIC_0000003.jpg')

    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=1)

    # ===== config.py 必需参数 =====
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', default='part')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--accumulation-steps', type=int, default=None)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', default='O1')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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