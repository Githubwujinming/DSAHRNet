"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import ntpath
from thop import profile,clever_format


def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor


def save_images(images, img_dir, name):
    """save images in img_dir, with name
    iamges: torch.float, B*C*H*W
    img_dir: str
    name: list [str]
    """
    for i, image in enumerate(images):
        print(image.shape)
        image_numpy = tensor2im(image.unsqueeze(0),normalize=False)
        basename = os.path.basename(name[i])
        print('name:', basename)
        save_path = os.path.join(img_dir,basename)
        save_image(image_numpy,save_path)


def save_visuals(visuals,img_dir,name):
    """
    """
    name = ntpath.basename(name)
    name = name.split(".")[0]
    print(name)
    # save images to the disk
    for label, image in visuals.items():
        if 'comp' in label:
            norm = False
            scale = 1
        elif 'L' in label :
            scale = 255
            norm = False
        else:
            scale = 255
            norm = True
        image_numpy = tensor2im(image,scale=scale,normalize=norm)
        img_path = os.path.join(img_dir, '%s_%s.png' % (name, label))
        save_image(image_numpy, img_path)


def get_params_size(model):
    num_params = 0
    for param in model.parameters():
            num_params += param.numel()
    return num_params / 1e6

def tensor2im(input_image, imtype=np.uint8,scale = 255, normalize=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if normalize:
            image_numpy = (image_numpy*std  + mean)  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return (image_numpy*scale).astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_by_path(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict,strict=False)
    return model

def get_model_complexity_info(model, use_cuda):
    x1 = torch.randn(1,3,256,256)
    x2 = torch.randn(1,3,256,256)
    if use_cuda:
        x1 = x1.cuda()
        x2 = x2.cuda()
        model = model.cuda(0)
    macs, params = profile(model, inputs=(x1,x2))
    macs, params = clever_format([macs, params], "%.3f")
    return macs,params