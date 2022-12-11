
import random
from re import L
from data.base_dataset import BaseDataset, get_albumentations, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import os
import cv2
import torchvision.transforms as transforms
import numpy as np


class ChangeDetectionDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    datafolder-tree
    dataroot:.
            ├─A
            ├─B
            ├─label
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        folder_A = 'A'
        folder_B = 'B'
        folder_L = 'label'
        self.istest = False
        if opt.phase == 'test':
            self.istest = True
        self.A_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_A), opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_B), opt.max_dataset_size))
        if not self.istest:
            self.L_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_L), opt.max_dataset_size))

        # print(self.A_paths)
    def _get_item_ori(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        transform_params = get_params(self.opt, A_img.size, test=self.istest)
        # apply the same transform to A B L
        transform = get_transform(self.opt, transform_params, test=self.istest)
        A = transform(A_img)
        B = transform(B_img)

        if self.istest:
            return {'A': A, 'A_paths': A_path, 'B': B, 'B_paths': B_path}

        L_path = self.L_paths[index]
        tmp = np.array(Image.open(L_path), dtype=np.uint32)/255
        L_img = Image.fromarray(tmp)
        transform_L = get_transform(self.opt, transform_params, method=transforms.InterpolationMode.NEAREST, normalize=False,
                                    test=self.istest)
        
        L = transform_L(L_img)
        if random.random() > 0.4:
            item = {'A': A, 'A_paths': A_path,
                'B': B, 'B_paths': B_path,
                'L': L, 'L_paths': L_path}
        else:
            item = {'A': B, 'A_paths': B_path,
                'B': A, 'B_paths': A_path,
                'L': L, 'L_paths': L_path}

    def _get_item_alb(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = cv2.cvtColor(cv2.imread(A_path),cv2.COLOR_BGR2RGB)
        B_img = cv2.cvtColor(cv2.imread(B_path),cv2.COLOR_BGR2RGB)
        L_path = self.L_paths[index]
        L_img = cv2.imread(L_path, cv2.IMREAD_GRAYSCALE)//255

        transform = get_albumentations(self.opt, test=self.istest)
        transformed = transform(image=A_img,imageB=B_img,mask=L_img)
        if self.istest:
            return {'A': transformed['image'], 'A_paths': A_path, 'B': transformed['imageB'], 'B_paths': B_path}
        if random.random() > 0.4:
            item = {'A': transformed['image'], 'A_paths': A_path,
                'B': transformed['imageB'], 'B_paths': B_path,
                'L': transformed['mask'][None,:], 'L_paths': L_path}
        else:
            item = {'A': transformed['imageB'], 'A_paths': B_path,
                'B': transformed['image'], 'B_paths': A_path,
                'L': transformed['mask'][None,:], 'L_paths': L_path}
        return item
    def __getitem__(self, index):
        # return self._get_item_ori(index)
        return self._get_item_alb(index)
# 
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
