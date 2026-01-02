import torch
from torchvision import transforms
from swinunet.networks.vision_transformer import SwinUnet as ViT_seg
from swinunet.config import get_config
import cv2
import numpy as np
import torchvision.utils as vutils
import torch.nn.functional as F
# ---------- 主流程 ----------
def swinmask(args,imgs):
    # ---------- Swin-Unet ----------
    config = get_config(args)
    swin_model = ViT_seg(config,
                    img_size=args.swin_img_size,
                    num_classes=args.num_classes).cuda()
    checkpoint = torch.load(args.isiccheckpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        swin_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        swin_model.load_state_dict(checkpoint)
    transform = transforms.Compose(
        [transforms.Resize((args.swin_img_size, args.swin_img_size))]
    )
    imgs = transform(imgs).cuda()


    with torch.no_grad():
        swin_pred = swin_model(imgs)  
        swin_pred = F.interpolate(
            swin_pred,               
            size=(256, 256),
            mode='nearest',           
            align_corners=None        
        )
        
        # # 步骤1：处理掩码张量（sigmoid + 阈值分割，全程保持张量格式）
        # # swin_pred = torch.sigmoid(swin_pred)  # 映射到0-1，shape: (2,1,224,224)
        # # swin_pred = (swin_pred > 0.3).float()  # 阈值分割，0/1浮点张量（vutils兼容float）
        # vutils.save_image(
        #     swin_pred,  # 输入张量：(2,1,224,224)
        #     "/home/liuyuxiu/models/Medical-SAM-Adapter/swinunet/1_swin_mask_grid.png",
        #     nrow=2,
        #     normalize=False, 
        #     padding=2  
        # )
        
        # vutils.save_image(
        #     imgs,  # 输入张量：(2,3,224,224)
        #     "/home/liuyuxiu/models/Medical-SAM-Adapter/swinunet/2_origin_img_grid.png",
        #     nrow=2, 
        #     normalize=True,  
        #     padding=2
        # )
 
    return swin_pred