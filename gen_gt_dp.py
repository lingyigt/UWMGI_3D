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

def load_3d_volume_img(img_path_list):
    '''img_path_list: a list of paths for all slices in one scan'''
    new_3d_imgs = []
    for i in range(len(img_path_list)):
        curr_img = cv2.imread(img_path_list[i], cv2.IMREAD_UNCHANGED)
        new_3d_imgs.append(curr_img)
    new_3d_imgs = np.stack(new_3d_imgs, axis=2).astype('float32')

    return new_3d_imgs
def load_3d_volume_msk(msk_path_list):
    '''img_path_list: a list of paths for all slices in one scan'''
    new_3d_msks = []
    for i in range(len(msk_path_list)):
        curr_msk = np.load(msk_path_list[i])
        new_3d_msks.append(curr_msk)
    new_3d_msks = np.stack(new_3d_msks, axis=3).astype('float32')
    return new_3d_msks

def get_dp(input, CFG):  # input dimension [d, h, w]: get 3d distance map for one class input needs to be on cpu to perform ndimage.distance_transform_edt
    h_resize, w_resize, d_resize = CFG.img_size
    dp_inside = ndimage.distance_transform_edt(input, sampling=[1/d_resize, 1/h_resize, 1/w_resize]) #, sampling=[1/180, 1/360, 1/360]) # normalize
    input_inverse = np.copy(input)
    input_inverse[input_inverse==1] = -1
    input_inverse[input_inverse==0] = 1
    input_inverse[input_inverse == -1] = 0
    dp_outside = ndimage.distance_transform_edt(input_inverse, sampling=[1/d_resize, 1/h_resize, 1/w_resize])
    dp = dp_inside + dp_outside
    return dp

if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 1110  # birthday
        num_worker = 16 # debug => 0  # LZ: how to set for only one GPU?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img_size = [266, 266, 144]
        debug = False

    set_seed(CFG.seed)

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
    df = df.drop(columns=['mask_path', 'image_path'])
    df = df.merge(df4, on='3d_id')
    df = df.merge(df5, on='3d_id')
    df = df.drop(columns=['id', 'slice'])
    if CFG.debug:
        df = df[10:]
    h_resize, w_resize, d_resize = CFG.img_size
    mask_paths = df['mask_path'].tolist()
    img_paths = df['image_path'].tolist()
    for index in range(len(mask_paths)):

        #get zoomed 3d
        img_path = img_paths[index]
        img = load_3d_volume_img(img_path)
        h, w, d = img.shape
        padding_depth = int((d_resize - d) / 2)
        img = zoom(img, (h_resize/h, w_resize/w, 1)) #TODO: check resized img
        img = np.pad(img, ((0, 0), (0, 0), (padding_depth, padding_depth)))
        if np.max(img)>0:
            img/=np.max(img)   #normalize img
        img = np.transpose(img, (2, 0, 1))  # [h, w, d] => [d, h, w]
        img = np.expand_dims(img, 0)

        img_path_splited = img_path[0].split('/')[:6]
        img_path_splited.append('img3d')
        img_3dpath = os.path.join(img_path_splited[0], img_path_splited[1], img_path_splited[2], img_path_splited[3],
                               img_path_splited[4], img_path_splited[5], img_path_splited[6])

        if not os.path.exists(img_3dpath):
            os.makedirs(img_3dpath)
        filenameimg3d = os.path.join(img_3dpath, 'img3d.npy')
        np.save(filenameimg3d, img)


        mask_path = mask_paths[index]
        mask = load_3d_volume_msk(mask_path)
        mask /= 255.0  # scale mask to [0, 1]  [h, w, c, d] c for number of classes
        mask = zoom(mask, (h_resize/h, w_resize/ w, 1, 1))     #TODO: save resized mask
        mask = np.pad(mask, ((0, 0), (0, 0), (0, 0), (padding_depth, padding_depth)))
        if np.max(mask)>0:
            mask/=np.max(mask)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = np.transpose(mask, (2, 3, 0, 1))  # [h, w, c, d] => [c, d, h, w]
        dp = np.zeros_like(mask)
        for class_num in range(mask.shape[0]):
            dp[class_num, :, :, :] = get_dp(mask[class_num, :, :, :], CFG)  # input dimension [d, h, w]:

        mask_path_splited = mask_path[0].split('/')[:8]
        mask_path_splited.append('dp')
        dp_path = os.path.join(mask_path_splited[0], mask_path_splited[1], mask_path_splited[2], mask_path_splited[3],
                                mask_path_splited[4], mask_path_splited[5], mask_path_splited[6], mask_path_splited[7],
                                mask_path_splited[8])

        mask_path_splited = mask_path[0].split('/')[:8]
        mask_path_splited.append('3d')
        mask_3dpath = os.path.join(mask_path_splited[0], mask_path_splited[1], mask_path_splited[2], mask_path_splited[3],
                               mask_path_splited[4], mask_path_splited[5], mask_path_splited[6], mask_path_splited[7],
                               mask_path_splited[8])

        if not os.path.exists(dp_path):
            os.makedirs(dp_path)
        if not os.path.exists(mask_3dpath):
            os.makedirs(mask_3dpath)
        filenamedp = os.path.join(dp_path, 'dp.npy')
        np.save(filenamedp, dp)
        filename3dmsk = os.path.join(mask_3dpath,'mask3d.npy')
        np.save(filename3dmsk, mask)
        print(str(index) + ' out of ', str(len(mask_paths)), ' finished')