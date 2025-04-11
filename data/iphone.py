import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle
import re
from . import base
import camera
from util import log,debug
from torch.utils.data._utils.collate import default_collate
class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 2160,3840
        super().__init__(opt,split)
        self.root = opt.data.root or "data/iphone"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        self.path_image = "{}".format(self.path)
        print(os.listdir(self.path_image))
        self.list = sorted(os.listdir(self.path_image),key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else float('inf'))
        # manually split train/val subsets
        num_val_split = int(len(self)*opt.data.val_ratio)
        self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        if subset: self.list = self.list[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt):
        # poses are unknown, so just return some dummy poses (identity transform)
        return camera.pose(t=torch.zeros(len(self),3))

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=torch.tensor(idx))
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    """def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image"""
    
    def get_image(self, opt, idx):
        image_fname = "{}/{}".format(self.path_image, self.list[idx])
        image = imageio.imread(image_fname, mode="F")  # Load as grayscale (H, W) with float values
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
        image = np.expand_dims(image, axis=-1)  # Convert to (H, W, 1)
        image = np.repeat(image, 3, axis=-1)  # Fake RGB by repeating channels -> (H, W, 3)
        image = PIL.Image.fromarray((image * 255).astype(np.uint8))  # Convert back to PIL Image
        return image

    def get_camera(self,opt,idx):
        self.focal = 23/(1.098*10**(-3))
        # fx = F/px , px= 1936*5.86*10**-3, 
        #fx = 8/(1936*5.86*10**-3)
        #fy=F/py, py = 1216*5.86*10**-3 fy = 8/(1216*5.86*10**-3 )
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose = camera.pose(t=torch.zeros(3)) # dummy pose, won't be used
        return intr,pose
    def get_camera(self,opt,idx):
        self.focal = self.raw_W*4.2/(12.8/2.55)
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose = camera.pose(t=torch.zeros(3)) # dummy pose, won't be used
        return intr,pose
