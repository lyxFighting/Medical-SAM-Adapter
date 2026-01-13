from logging import config
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from glob import glob

from utils import *


from .atlas import Atlas
from .brat import Brat
from .ddti import DDTI
# from .isic import ISIC2016
from .generaldataset import dataset
from .kits import KITS
from .lidc import LIDC
from .lnq import LNQ
from .pendal import Pendal
from .refuge import REFUGE
from .segrap import SegRap
from .stare import STARE
from .toothfairy import ToothFairy
from .wbc import WBC


def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])

    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])


    img_ext = '.png'
    mask_ext = '.png'      
    if args.dataset == 'ISIC':
        img_ext = '.jpg'
        mask_ext = '_Segmentation.png'
    elif args.dataset == 'busi':
        mask_ext = '_mask.png'    
    
    img_ids = sorted(glob(os.path.join(args.data_path,'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.3, random_state=2981)
    isic_train_dataset = dataset(args, args.data_path, img_ids=train_img_ids, transform=transform_train, transform_msk=transform_train_seg,img_ext=img_ext,mask_ext=mask_ext)
    isic_test_dataset = dataset(args, args.data_path, img_ids=val_img_ids, transform = transform_test, transform_msk= transform_test_seg,img_ext=img_ext,mask_ext=mask_ext)
    nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
      
    return nice_train_loader, nice_test_loader