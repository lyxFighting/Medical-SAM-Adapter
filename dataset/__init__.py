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
isicdataset=1
if isicdataset==0:
    from .isic import ISIC2016
    print('[INFO] Using original ISIC transform pipeline')
elif isicdataset==1:
    from .isic_joint import ISIC2016, ISICJointTransform2D
    print('[INFO] Using ISICJointTransform2D')
else:
    from .isic_idgeneral import ISIC2016
    print('[INFO] Using ISIC2016 with click prompt')

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


    
    if args.dataset == 'isic':
        '''isic data'''
        if isicdataset==0:
            isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
            isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')
        elif isicdataset==1:
            joint_tf_train = ISICJointTransform2D(
                img_size=args.image_size,
                low_img_size=args.out_size,
                ori_size=args.image_size,
                p_flip=0.5,
                p_rota=0.5,
                p_scale=0.5,
                p_gaussn=0.5,
                p_contr=0.5,
                p_gama=0.5,
                color_jitter_params=(0.1, 0.1, 0.1, 0.1),
            )

            joint_tf_test = ISICJointTransform2D(
                img_size=args.image_size,
                low_img_size=args.out_size,
                ori_size=args.image_size,
            )
            isic_train_dataset = ISIC2016(args,args.data_path,joint_transform=joint_tf_train,mode='Training')
            isic_test_dataset = ISIC2016(args,args.data_path,joint_transform=joint_tf_test,mode='Test')
        else:
            img_ext = '.jpg'
            mask_ext = '_Segmentation.png'
            args.data_path = '/home/liuyuxiu/models/Medical-SAM-Adapter/data/ISIC/isic_all'
            img_ids = sorted(glob(os.path.join(args.data_path,'images', '*' + img_ext)))
            img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
            train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.3, random_state=2981)
            isic_train_dataset = ISIC2016(args, args.data_path, img_ids=train_img_ids, transform=transform_train, transform_msk=transform_train_seg,img_ext=img_ext,mask_ext=mask_ext)
            isic_test_dataset = ISIC2016(args, args.data_path, img_ids=val_img_ids, transform = transform_test, transform_msk= transform_test_seg,img_ext=img_ext,mask_ext=mask_ext)

        nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'decathlon':
        nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)


    elif args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'LIDC':
        '''LIDC data'''
        # dataset = LIDC(data_path = args.data_path)
        dataset = MyLIDC(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'DDTI':
        '''DDTI data'''
        refuge_train_dataset = DDTI(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        refuge_test_dataset = DDTI(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'Brat':
        '''Brat data'''
        dataset = Brat(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'STARE':
        '''STARE data'''
        # dataset = LIDC(data_path = args.data_path)
        dataset = STARE(args, data_path = args.data_path, transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'kits':
        '''kits data'''
        dataset = KITS(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'WBC':
        '''WBC data'''
        dataset = WBC(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'segrap':
        '''segrap data'''
        dataset = SegRap(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'toothfairy':
        '''toothfairy data'''
        dataset = ToothFairy(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'atlas':
        '''atlas data'''
        dataset = Atlas(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'pendal':
        '''pendal data'''
        dataset = Pendal(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'lnq':
        '''lnq data'''
        dataset = LNQ(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader