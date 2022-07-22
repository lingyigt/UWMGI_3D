###############################################################
##### @Title:  UWMGI baseline
##### @Time:  2022/6/3
##### @Author: frank
##### @Describe: 
        #  part0: data preprocess
        #  part1: build_transforme() & build_dataset() & build_dataloader()
        #  part2: build_model()
        #  part3: build_loss()
        #  part4: build_metric()
        #  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
###############################################################
import os
import pdb
from tkinter.messagebox import NO
import cv2
import time
import glob
import random

from cv2 import transform
import cupy as cp # https://cupy.dev/ => pip install cupy-cuda102
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html

from sklearn.model_selection import StratifiedGroupKFold # Sklearn
import albumentations as A # Augmentations
import segmentation_models_pytorch as smp # smp
from sklearn.model_selection import KFold
from pytorch3dunet.unet3d.model import UNet3D
from pytorch3dunet.unet3d.losses import DiceLoss
from pytorch3dunet.unet3d.metrics import DiceCoefficient
from pytorch3dunet.augment.transforms import ElasticDeformation
from pytorch3dunet.augment.transforms import GaussianBlur3D
from pytorch3dunet.augment.transforms import Compose
from scipy import ndimage
from models.unet import UNet
from scipy.ndimage import zoom
from numba import jit
from matplotlib import pyplot as plt
def set_seed(seed=1110):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################
##### part0: data preprocess
###############################################################
def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row

def path2info(row):
    path = row['image_path']
    path = path.replace('\\','/') # LZ: To solve the issue for path in windows
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    # row['id'] = f'case{case}_day{day}_slice_{slice_}'
    return row

def mask2rle(msk, thr=0.5):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk    = cp.array(msk)
    pixels = msk.flatten()
    pad    = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs   = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def masks2rles(msks, ids, heights, widths):
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx in range(msks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        msk = cv2.resize(msks[idx], 
                        dsize=(width, height), 
                        interpolation=cv2.INTER_NEAREST) # back to original shape
        rle = [None]*3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[...,midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]]*len(rle))
        pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    return pred_strings, pred_ids, pred_classes

###############################################################
##### part1: build_transforms & build_dataset & build_dataloader
###############################################################
''' Below is 2D transforms
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            A.OneOf([
                A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            ], p=1),

            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),#LZ: Randomly apply affine transforms: translate, scale and rotate the input.
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5), #LZ: CoarseDropout of the rectangular regions in the image.
            ], p=1.0),
        
        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
        }
    return data_transforms

'''
def build_transforms(CFG):
    data_transforms = {
        "train": Compose([ElasticDeformation(random_state=np.random.RandomState(CFG.seed), spline_order=0,
                                             alpha=2000, sigma=50, execution_probability=0.5, apply_3d=True),
                          GaussianBlur3D(sigma=[.1, 2.], execution_probability=0.5)]),

        }
    return data_transforms


class build_dataset(Dataset):
    def __init__(self, df, label='train', transforms=None, cfg=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist() # image
        self.ids = df['3d_id'].tolist()

        if 'mask_path' in df.columns:
            self.mask_paths = df['mask_path'].tolist() # mask
        else:
            self.mask_paths = None

        if 'mask_dp_path' in df.columns:
            self.dp_paths = df['mask_dp_path'].tolist()
        else:
            self.dp_paths = None
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        #### load id
        id = self.ids[index]   # This id is 3D id
        #### load image
        img_path = self.img_paths[index]
        img = self.load_3d_volume(img_path)   # This is 3D images (including slice from one case on one certain day)
        h, w, d = img.shape
        padding_height = int((360-h)/2)
        padding_width = int((360-w)/2)
        padding_depth = int((180-d)/2)
        #img = np.pad(img,((padding_height,padding_height),(padding_width,padding_width),(padding_depth,padding_depth))) #[h, w, d]
        img = zoom(img, (244/h, 244/w, 1))  #144/d))  #use resizing instead of padding
        img = np.pad(img,((0,0),(0,0),(padding_depth,padding_depth))) #[h, w, d]
        if np.max(img)>0:
            img/=np.max(img)
        if self.label == 'train': # train
            #### load mask
            mask_path = self.mask_paths[index]
            mask = self.load_3d_volume_msk(mask_path)
            mask/=255.0 # scale mask to [0, 1]  [h, w, c, d] c for number of classes
            #mask = np.pad(mask, ((padding_height, padding_height), (padding_width, padding_width), (0, 0), (padding_depth, padding_depth)))
            mask = zoom(mask, (244/h, 244/w, 1, 1)) #144/d))   #use resizing instead of padding
            mask = np.pad(mask, ((0, 0), (0, 0), (0, 0), (padding_depth, padding_depth)))

            if np.max(mask)>0:
                mask/=np.max(mask)
            mask[mask<0.5]=0
            mask[mask>=0.5]=1
            img = np.transpose(img, (2, 0, 1)) # [h, w, d] => [d, h, w]
            img = np.expand_dims(img, 0)   #Add channel dimension: [d,h,w] => [c,d,h,w], c = 1 i.e., input size
            mask = np.transpose(mask, (2, 3, 0, 1)) # [h, w, c, d] => [c, d, h, w]

            ### load distance map
            dp_path = self.dp_paths[index]
            distance_map = np.load(dp_path)
            mask_w_dp = np.concatenate((mask, distance_map))
            return torch.tensor(img), torch.tensor(mask_w_dp)
        
        elif self.label == 'val': # test
            mask_path = self.mask_paths[index]
            mask = self.load_3d_volume_msk(mask_path)
            mask /= 255.0  # scale mask to [0, 1]  [h, w, c, d] c for number of classes
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            #mask = np.pad(mask, (
            #(padding_height, padding_height), (padding_width, padding_width), (0, 0), (padding_depth, padding_depth)))
            mask = zoom(mask, (244 / h, 244 / w, 1, 1))  # 144/d))   #use resizing instead of padding
            mask = np.pad(mask, ((0, 0), (0, 0), (0, 0), (padding_depth, padding_depth)))
            if np.max(mask)>0:
                mask/=np.max(mask)
            mask[mask<0.5]=0
            mask[mask>=0.5]=1
            img = np.transpose(img, (2, 0, 1))  # [h, w, d] => [d, h, w]
            img = np.expand_dims(img, 0)  # Add channel dimension: [d,h,w] => [c,d,h,w], c = 1 i.e., input size
            mask = np.transpose(mask, (2, 3, 0, 1))  # [h, w, c, d] => [c, d, h, w]
            ### load distance map
            dp_path = self.dp_paths[index]
            distance_map = np.load(dp_path)
            mask_w_dp = np.concatenate((mask, distance_map))
            return torch.tensor(img), torch.tensor(mask_w_dp)

        else:  # test  #TODO:Add padding and transform here, but later
            ### augmentations
            data = self.transforms(image=img)
            img  = data['image']
            img = np.transpose(img, (2, 0, 1)) # [h, w, c] => [c, h, w]
            return torch.tensor(img), id, h, w

    ###############################################################
    ##### >>>>>>> construct 3d volume data <<<<<<
    ###############################################################

    def load_3d_volume(self, img_path_list):
        '''img_path_list: a list of paths for all slices in one scan'''
        new_3d_imgs = []
        for i in range(len(img_path_list)):
            curr_img = cv2.imread(img_path_list[i], cv2.IMREAD_UNCHANGED)
            new_3d_imgs.append(curr_img)
        new_3d_imgs = np.stack(new_3d_imgs, axis=2).astype('float32')


        return new_3d_imgs

    def load_3d_volume_msk(self, msk_path_list):
        '''img_path_list: a list of paths for all slices in one scan'''
        new_3d_msks = []
        for i in range(len(msk_path_list)):
            curr_msk = np.load(msk_path_list[i])
            new_3d_msks.append(curr_msk)
        new_3d_msks = np.stack(new_3d_msks, axis=3).astype('float32')
        return new_3d_msks

def build_dataloader(df, fold, data_transforms, CFG):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    train_dataset = build_dataset(train_df, label='train', transforms=data_transforms['train'], cfg=CFG)
    valid_dataset = build_dataset(valid_df, label='val', transforms=data_transforms['train'], cfg=CFG)

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=CFG.num_worker, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=CFG.num_worker, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
def build_model(CFG, test_flag=False):
    model = UNet3D(in_channels=1, out_channels=3, final_sigmoid=True, f_maps=16, layer_order='gcr', num_groups=1,
                          num_levels=4, is_segmentation=True, conv_padding=1)

    model.to(CFG.device)
    return model

###############################################################
##### >>>>>>> part3: build_loss <<<<<<
###############################################################
def build_loss():
    BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
    dice_loss = DiceLoss()
    return {"BCELoss":BCELoss, "TverskyLoss":TverskyLoss, "3DdiceLoss": dice_loss}

###############################################################
##### >>>>>>> part4: build_metric <<<<<<
###############################################################
'''2D dice
def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice
'''

# Metric: 3D dice



def dice_coef(y_true,y_pred,smooth=0.0001):  # INPUT:[b, c, d, h, w]
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    total_dice = 0
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            curr_class_pred = y_pred[i, j, :, :, :]
            curr_class_gt = y_true[i, j, :, :, :]
            intersection = torch.dot(curr_class_pred.reshape(-1), curr_class_gt.reshape(-1))
            union = torch.sum(curr_class_pred.reshape(-1)) + torch.sum(curr_class_gt.reshape(-1))
            total_dice = total_dice + (2*intersection + smooth)/(union + smooth)
    total_dice = total_dice/(y_pred.shape[0]*y_pred.shape[1])
    return total_dice
#def dice_coef(y_true,y_pred):
#    dice = DiceCoefficient()
#    return dice(y_true, y_pred)
# Metric: 3D HD
def HD_coeff(y_true,y_pred,CFG):  # Note: y_pred should be thresholded
    total_hd_coeff = 0
    y_true_cpu = y_true.cpu().detach().numpy()
    y_pred_cpu = y_pred.cpu().detach().numpy()
    hd_total = 0
    for i in range(y_true.shape[0]):
        for j in range(CFG.num_classes):
            dp_y_true = y_true_cpu[i, j+3, :, :, :]
            dp_y_pred = get_dp(y_pred_cpu[i, j, :, :, :])
            dp_y_true = dp_y_true/np.sqrt(3)
            dp_y_pred = dp_y_pred/np.sqrt(3)
            diff_pred_true = np.absolute(y_pred_cpu[i, j, :, :]-y_true_cpu[i, j, :, :])
            hd_true_pred = np.max(np.multiply(diff_pred_true, dp_y_pred))
            hd_pred_true = np.max(np.multiply(diff_pred_true, dp_y_true))
            hd = max(hd_true_pred, hd_pred_true)
            hd_total += hd
    hd_ave = hd_total/(y_true.shape[0]*CFG.num_classes)
    hd_ave_normalize = hd_ave
    return hd_ave_normalize
def get_dp(input):  # input dimension [d, h, w]: get 3d distance map for one class input needs to be on cpu to perform ndimage.distance_transform_edt
    dp_inside = ndimage.distance_transform_edt(input, sampling=[1/180, 1/244, 1/244]) #, sampling=[1/180, 1/360, 1/360]) # normalize
    input_inverse = np.copy(input)
    input_inverse[input_inverse==1] = -1
    input_inverse[input_inverse==0] = 1
    input_inverse[input_inverse == -1] = 0
    dp_outside = ndimage.distance_transform_edt(input_inverse, sampling=[1/180, 1/244, 1/244])
    dp = dp_inside + dp_outside
    return dp

# 3D HD loss
def hd_loss(y_true, y_pred, CFG):
    y_pred = y_pred.to(torch.float32)
    y_true = y_true.to(torch.float32)
    total_hd_loss = 0
    for i in range(y_true.shape[0]):  # iterate through batch
        for j in range(CFG.num_classes):   # iterate through class
            curr_class_dp = y_true[i, j+3, :, :, :]
            curr_class_gt = y_true[i, j, :, :, :]
            curr_class_pred = y_pred[i, j, :, :, :]
            curr_class_hdloss_map = torch.mul(torch.square(curr_class_gt - curr_class_pred), torch.pow(curr_class_dp, 2))
            curr_class_hdloss = torch.sum(curr_class_hdloss_map)/(y_true.shape[2]*y_true.shape[3]*y_true.shape[4])
            total_hd_loss += curr_class_hdloss

    ave_hd_loss = total_hd_loss/(y_true.shape[0]*CFG.num_classes)
    return ave_hd_loss
def dice_loss(y_true,y_pred,smooth=0.0001):  # INPUT:[b, c, d, h, w]
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    total_dice = 0
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            curr_class_pred = y_pred[i, j, :, :, :]
            curr_class_gt = y_true[i, j, :, :, :]
            intersection = torch.dot(curr_class_pred.reshape(-1), curr_class_gt.reshape(-1))
            union = torch.sum(curr_class_pred.reshape(-1)) + torch.sum(curr_class_gt.reshape(-1))
            total_dice = total_dice + (2*intersection + smooth)/(union + smooth)
    total_dice = total_dice/(y_pred.shape[0]*y_pred.shape[1])
    loss = 1 - total_dice
    return loss
###############################################################
##### >>>>>>> part5: train & validation & test <<<<<<
###############################################################
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG, epoch):
    model.train()
    scaler = amp.GradScaler() 
    losses_all, dice_all, hd_all = 0, 0, 0
    loss_ratio = 20
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, (images, masks) in pbar:
        optimizer.zero_grad()

        images = images.to(CFG.device, dtype=torch.float) # [b, c, d, h, w]
        masks  = masks.to(CFG.device, dtype=torch.float)  # [b, c, d, h. w]

        with amp.autocast(enabled=True):
            y_preds = model(images) # [b, c, d, h, w]
            y_preds = torch.sigmoid(y_preds)
            ave_dice_loss = dice_loss(masks[:, :3, :, :, :], y_preds)   #argument order: (input, target) as written in the source code
            ave_hd_loss = hd_loss(masks, y_preds, CFG)           #bce_loss = losses_dict["BCELoss"](y_preds, masks)
            #ave_hd_loss = 0
            #tverskly_loss = losses_dict["TverskyLoss"](y_preds, masks)
            #losses = bce_loss + tverskly_loss
            #losses = 0.8*dice_loss + ave_hd_loss

            #if epoch < 10:
            if epoch >100:
                losses = dice_loss
            else:  # Add hd loss after 10 epochs
                losses = 100*ave_hd_loss + ave_dice_loss
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses_all += losses.item()/images.shape[0]  #losses_all is the sum of all losses along one epoch i.e., ave_dice per iter*iter per epoch
        dice_all += ave_dice_loss.item()/images.shape[0]
        hd_all += ave_hd_loss.item()/images.shape[0]
       #hd_all += ave_hd_loss/images.shape[0]


        #dice_all += dice_loss/images.shape[0]
        #hd_all += ave_hd_loss/images.shape[0]
        loss_ratio = hd_all/dice_all

    
    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.6f}".format(current_lr), flush=True)
    print("loss: {:.6f}, hd_all: {:.6f}, dice_all: {:.6f}".format(losses_all, hd_all, dice_all), flush=True)
    return losses_all #TODO: for the REDUCELRONPLATEAU lr scheduler
@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    val_scores = []
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, masks) in pbar:
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, d, w, h]
        masks   = masks.to(CFG.device, dtype=torch.float)  # [b, c, d, w, h]
        # y_preds is already normalized with sigmoid in model function
        y_preds = model(images) 
        y_preds[y_preds > CFG.thr] = 1
        y_preds[y_preds < CFG.thr] = 0
        val_dice = dice_coef(masks[:, :3, :, :, :], y_preds).cpu().detach().numpy()   #TODO:check the rightfulness of this function
        val_hd = HD_coeff(masks, y_preds, CFG)  # Transfer to cpu inside the function
        val_scores.append([val_dice, val_hd])

    val_scores = np.mean(val_scores, axis=0)
    val_dice, val_hd = val_scores
    print("val_dice: %8.6f, val_hd: %8.6f" % (val_dice, val_hd), flush=True)
    total_score = 0.4*val_dice + 0.6*(1-val_hd)
    print("total_score: %8.6f" % total_score)
    return total_score

@torch.no_grad()
def test_one_epoch(ckpt_paths, test_loader, CFG):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, ids, h, w) in pbar:

        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        size = images.size()
        masks = torch.zeros((size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32) # [b, c, w, h]
        
        ############################################
        ##### >>>>>>> cross validation infer <<<<<<
        ############################################
        for sub_ckpt_path in ckpt_paths:
            model = build_model(CFG, test_flag=True)
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            y_preds = model(images) # [b, c, w, h]
            y_preds   = torch.nn.Sigmoid()(y_preds)    #This will be a 3D prediction with depth dimension later
            # Need to convert y_preds into 2D slices, and keep following code same should be OK, MAY need iteration to extract slice

            masks += y_preds/len(ckpt_paths)
        
        masks = (masks.permute((0, 2, 3, 1))>CFG.thr).to(torch.uint8).cpu().detach().numpy() # [n, h, w, c]
        result = masks2rles(masks, ids, h, w)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
    return pred_strings, pred_ids, pred_classes


if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 1110  # birthday
        num_worker = 16 # debug => 0  # LZ: how to set for only one GPU?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_fold = "ckpt-liz-20220630-code3D"
        ckpt_name = "img244244180_bs1_fold1-fm16-hdloss-adjustlossratio-load-dp-fullset"
        
        # step2: data
        n_25d_shift = 2   #LZ: no need to generate 2.5 D data seperately, all integrated into one
        n_fold = 5
        img_size = [244, 244, 180]
        train_bs = 1
        valid_bs = 2

        # step3: model
        num_classes = 3

        # step4: optimizer
        epoch = 80
        lr = 1e-3
        wd = 0.01  #1e-5  # TODO:try default value 0.01
        lr_drop = 8   # use 0.1*previous learning rate per 8 epochs

        # step5: infer
        thr = 0.6

        # Other parameters
        one_fold_train = True
        debug = False  # Running only 3000 data on my laptop
    set_seed(CFG.seed)
    ckpt_path = f"../input/{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    train_val_flag = True
    if train_val_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        # document: https://pandas.pydata.org/docs/reference/frame.html
        df = pd.read_csv('../input/uwmgi-mask-dataset/train.csv')
        df['segmentation'] = df.segmentation.fillna('') # .fillna(): 填充NaN的值为空
        # rle mask length
        df['rle_len'] = df.segmentation.map(len) # .map(): 特定列中的每一个元素应用一个函数len
        # image/mask path
        df['image_path'] = df.image_path.str.replace('/kaggle/','../') # .str: 特定列应用python字符串处理方法 
        df['mask_path'] = df.mask_path.str.replace('/kaggle/','../')
        df['mask_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy')
        # rle list of each id


        df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # .grouby(): 特定列划分group.
        # total length of all rles of each id
        df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # .agg(): 特定列应用operations

        df = df.drop(columns=['segmentation', 'class', 'rle_len']) # .drop(): 特定列的删除
        df = df.groupby(['id']).head(1).reset_index(drop=True)

        df3 = df.id.to_frame()
        df3['3d_id_0'] = df3.id.str.split('_')
        df3['3d_id'] = df3['3d_id_0'].str[0] + '_' + df3['3d_id_0'].str[1]
        df3 = df3.drop(columns=['3d_id_0'])
        df = df.merge(df3, on=['id'])

        df4 = df.groupby(['3d_id'])['image_path'].agg(list).to_frame().reset_index()
        df5 = df.groupby(['3d_id'])['mask_path'].agg(list).to_frame().reset_index()
        df = df.groupby(['3d_id']).head(1).reset_index(drop=True)
        df = df.drop(columns=['mask_path','image_path'])
        df = df.merge(df4, on='3d_id')
        df = df.merge(df5, on='3d_id')
        df = df.drop(columns=['id','slice'])

        df_mask_dp = df[['3d_id', 'mask_path']]
        mask_path = df.mask_path.to_frame()
        df_mask_dp['mask_path_splited'] = df_mask_dp.mask_path.map(lambda x: x[0])
        df_mask_dp['prefix'] = df_mask_dp.mask_path_splited.str.split('scan')
        df_mask_dp['prefix'] = df_mask_dp['prefix'].map(lambda x:x[0] + '/dp/dp.npy')
        df_mask_dp['mask_dp_path'] = df_mask_dp.prefix.str.replace('//', '/')  # The path for distance map
        df_mask_dp = df_mask_dp.drop(columns = ['mask_path', 'mask_path_splited', 'prefix'])
        df = df.merge(df_mask_dp, on='3d_id')

        # LZ: Debug on my laptop:
        if CFG.debug:
            df = df[:50]

        ###############################################################
        ##### >>>>>>> trick1: cross validation train <<<<<<
        ###############################################################
        # document: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        # No need to use stratifiedGroupKFold
        #skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        #for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold


        #for fold in range(CFG.n_fold):
        for fold in range(1):

            print(f'#'*80, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*80, flush=True)

            ###############################################################
            ##### >>>>>>> step2: combination <<<<<<
            ##### build_transforme() & build_dataset() & build_dataloader()
            ##### build_model() & build_loss()
            ###############################################################
            data_transforms = build_transforms(CFG)  
            train_loader, valid_loader = build_dataloader(df, fold, data_transforms, CFG) # dataset & dtaloader
            model = build_model(CFG) # model
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
            #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 5) #TODO: try this lr_scheduler as introduced in the HD paper
            losses_dict = build_loss() # loss

            best_val_score = 0
            best_epoch = 0
            
            for epoch in range(1, CFG.epoch+1):
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<
                ###############################################################
                #train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                #lr_scheduler.step()
                losses_all = train_one_epoch(model, train_loader, optimizer, losses_dict, CFG, epoch) #TODO: for the REDUCELRONPLATEAU lr scheduler
                lr_scheduler.step(losses_all) #TODO: for the REDUCELRONPLATEAU lr scheduler
                #val_dice, val_jaccard = valid_one_epoch(model, valid_loader, CFG)
                val_score = valid_one_epoch(model, valid_loader, CFG)

                ###############################################################
                ##### >>>>>>> step4: save best model <<<<<<
                ###############################################################
                is_best = (val_score > best_val_score)
                best_val_score = max(best_val_score, val_score)
                # LZ: only save the checkpoints which generate higher validation score
                if is_best:
                    save_path = f"{ckpt_path}/best_fold{fold}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path)
                    torch.save(model.state_dict(), save_path)

                epoch_time = time.time() - start_time
                print("epoch:%2d, time:%5.2fs, best:%6.4f\n" % (epoch, epoch_time, best_val_score), flush=True)


    test_flag = False
    if test_flag:
        set_seed(CFG.seed)
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
        if not len(sub_df):
            sub_firset = True
            sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')[:1000*3]
            sub_df = sub_df.drop(columns=['class','segmentation']).drop_duplicates()
            paths = glob(f'../input/uw-madison-gi-tract-image-segmentation/train/**/*png',recursive=True)
        else:
            sub_firset = False
            sub_df = sub_df.drop(columns=['class','predicted']).drop_duplicates()
            paths = glob(f'../input/uw-madison-gi-tract-image-segmentation/test/**/*png',recursive=True)
        sub_df = sub_df.apply(get_metadata,axis=1)
        path_df = pd.DataFrame(paths, columns=['image_path'])
        path_df = path_df.apply(path2info, axis=1)
        test_df = sub_df.merge(path_df, on=['case','day','slice'], how='left')

        data_transforms = build_transforms(CFG)
        test_dataset = build_dataset(test_df, label=False, transforms=data_transforms['valid_test'], cfg=CFG)
        test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=2, shuffle=False, pin_memory=False)

        ###############################################################
        ##### >>>>>>> step2: infer <<<<<<
        ###############################################################
        # attention: change the corresponding upload path to kaggle.
        ckpt_paths  = glob(f'{ckpt_path}/best*')
        assert len(ckpt_paths) == CFG.n_fold, "ckpt path error!"

        pred_strings, pred_ids, pred_classes = test_one_epoch(ckpt_paths, test_loader, CFG)

        ###############################################################
        ##### step3: submit
        ###############################################################
        pred_df = pd.DataFrame({
            "id":pred_ids,
            "class":pred_classes,
            "predicted":pred_strings
        })
        if not sub_firset:
            sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
            del sub_df['predicted']
        else:
            sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')[:1000*3]
            del sub_df['segmentation']
            
        sub_df = sub_df.merge(pred_df, on=['id','class'])
        sub_df.to_csv('submission.csv',index=False)