# -*- coding: utf-8 -*-
# date: 2022/3/15
# Project: BovwSfaCD-pytorch
# File Name: process.py
# Description: 把val数据处理成需要的形状 (256, 256)
# Author: Anefuer_kpl
# Email: 374774222@qq.com
# from https://github.com/kangpeilun/utils-for-img-process
from genericpath import exists
import glob
import os
from os.path import join
import PIL
from PIL import Image
import random
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# ============ config =============
IF_RESIZE = False          # 裁剪前是否对图像进行resize
RESIZE = 286
CROP = (256, 256)         # 裁剪的大小 对应 h, w
DIR_NAME = "../../datasets/BCD/LEVIR-CD/val/"      # 原数据存放的文件夹名称
NEW_DIR_NAME = '../../datasets/BCD/LEVIR-CD_cropped/val/'
CROP_NUM_EACH_PIC = 16     # 每张图片上随机裁剪多少张图片
RAND_CROP=False

SEED = 0                  # 随机种子，使得每次划分结果都一致
THREAD = 10               # 启用多线程划分数据, 线程数

random.seed(SEED)
# ============ config =============
DATA_DIR = join(os.path.dirname(os.path.realpath(__file__)), DIR_NAME)   # 获取 原始 数据存放文件夹完整路径
NEW_DIR = join(os.path.dirname(os.path.realpath(__file__)), NEW_DIR_NAME)   # 获取 新生成 数据存放文件夹完整路径


def get_img_size():
    '''获取待裁剪图片的尺寸，默认所有图片的尺寸大小都一致'''
    sub_dir = join(DATA_DIR, os.listdir(DATA_DIR)[0])   # 获取原数据文件夹第一个子文件夹
    img_file = join(sub_dir, os.listdir(sub_dir)[0])    # 获取 第一个文件
    img = Image.open(img_file)

    return img.size[0], img.size[1]


def check_dir(dir):
    '''检查文件夹是否存在，并创建不存在的文件夹'''
    if not os.path.exists(dir):
        os.mkdir(dir)


def make_dirs():
    '''
    保持新数据文件夹结构与原数据文件夹结构一致
    Args: 以我的数据文件为例，子文件夹的数量可以和我不一样，代码会自动处理
        DATA_DIR: 原数据存放文件夹结构
                data --|
                       A
                       B
                     label
        NEW_DIR: 新数据存放文件夹
            data-new --|
                       A
                       B
                     label
    Returns: 返回新数据 子文件夹 路径
    '''
    check_dir(NEW_DIR)  # 创建新数据文件夹
    ori_list_dir = [join(DATA_DIR, dir_name) for dir_name in os.listdir(DATA_DIR)]   # 获取原数据子文件夹完整路径
    new_dir_list = []  # 获取新数据子文件夹完整路径
    for dir in os.listdir(DATA_DIR):
        dir_sub = join(NEW_DIR, dir)  # 子文件夹完整路径
        new_dir_list.append(dir_sub)
        check_dir(dir_sub)  # 获取原始数据文件夹所有的 文件夹，并在新数据目录创建对应的文件夹

    return ori_list_dir, new_dir_list


def crop_one_img(file_path, save_path, crop_area):
    img = Image.open(file_path)
    if IF_RESIZE:   # 缩放
        img = img.resize((RESIZE, RESIZE), Image.BILINEAR)

    img = img.crop(crop_area)   # 获取裁剪后的图片
    img.save(save_path)


def process_data():
    ori_list_dir, new_dir_list = make_dirs()  # 获取新 原数据子文件夹完整路径
    h, w = get_img_size()   # 获取图片尺寸
    bar = tqdm(range(1, CROP_NUM_EACH_PIC+1), total=CROP_NUM_EACH_PIC, ascii=True)
    list_x = range(0, w, CROP[0])
    list_y = range(0, h, CROP[1])
    rows = len(list_y)
    cols = len(list_x)
    for crop_num in bar:
        
        if IF_RESIZE:
            h, w = RESIZE, RESIZE       # 启用resize，修改图片的高 宽
        crop_range_h, crop_range_w = h - CROP[0], w - CROP[1]  # 获取可以进行裁剪的区域,防止裁剪时越界
        if RAND_CROP:
            x, y = random.randint(0, crop_range_h), random.randint(0, crop_range_w)  # 产生随机左上角裁剪坐标点
        else:
            row, col = (crop_num - 1)//rows, (crop_num - 1)%cols
            x, y = list_x[row],list_y[col]
        crop_area = (x, y, x + CROP[0], y + CROP[1])  # 获取被裁减区域坐标

        bar.set_description('{}\tCrop_size:{}\tCrop_area:{}'.format(NEW_DIR, CROP, crop_area))

        for idx, (ori_dir_path, new_dir_path) in enumerate(zip(ori_list_dir, new_dir_list)):  # 一组一组的数据文件夹进行裁剪
            '''所有的子文件夹中的数据都在 相同的 crop_area 中被裁剪一次，总共重复CROP_NUM_EACH_PIC次
                exa: A_ori--|      A_new
                       xxx.png
                       xxx.png
            '''
            with ThreadPoolExecutor(THREAD) as t:
                for file in os.listdir(ori_dir_path):
                    file_path = join(ori_dir_path, file)  # 原数据文件路径
                    save_path = join(new_dir_path, file.rsplit('.')[0]+f'_{crop_num}.png')
                    t.submit(crop_one_img, file_path=file_path, save_path=save_path, crop_area=crop_area)
                    # img = Image.open(file_path)
                    # cropImg = img.crop(crop_area)   # 获取裁剪后的图片
                    # cropImg.save(join(new_dir_path, file.rsplit('.')[0]+f'_{crop_num}.png'))

def rename(file_path,new_path):
    if not os.path.exists(file_path):
        print(file_path+" not exists")
        return
    os.rename(file_path, new_path)
        

def files_rename(dir,prefix='1_'):
    with ThreadPoolExecutor(THREAD) as t:
        for file in os.listdir(dir):
            new_name = prefix+'_'+file.split('_')[1]
            file_path = join(dir,file)
            new_path = join(dir,new_name)
            t.submit(rename, file_path=file_path, new_path=new_path)

def cut(img_dir, output_img,  patch_size_w, patch_size_h, stride_w=256, stride_h=256):
    """
    传入图片的路径，遍历切割;
    img_dir 为图片待切割图片路径；
    out_img:切割好的img的输出路径;
    output_npy:输出npy
    output_mat:输出mat数据的路径
    patch_size_w:切割的图像宽度
    patch_size_h:切割的图像的高度
    stride_w:横向移动大小
    stride_h:高度移动大小
    """
    img_dir = img_dir
    file_list = glob.glob(img_dir + '*.jpg')#遍历文件里的图片
    print(len(file_list))
    for i in range(len(file_list)):
        img = Image.open(file_list[i])
        # img = cv2.imread(),cv2生成的是numpy
        # AttributeError: 'numpy.ndarray' object has no attribute 'crop'
        # 如果是cv2，可以使用img.shape,
        # weight, height, channel = img.size
        weight, height = img.size
        # 偏移量
        stride_w = stride_w
        stride_h = stride_h
        # 切割开始点
        x1 = 0
        y1 = 0
        x2 = patch_size_w
        y2 = patch_size_h
        n = 0
        while x2 <= weight:
            while y2 <= height:
                crop_name = str(i) + str(n) + ".jpg"
                # 保存切割后的图片
                img2 = img.crop((y1, x1, y2, x2))
                img2.save(output_img + crop_name)
                n = n + 1
                # 切割移动方式为从上到下，再从左到右
                y1 = y1 + stride_h
                y2 = y1 + patch_size_h
            # 左右移动
            x1 = x1 + stride_w
            x2 = x1 + patch_size_w
            y1 = 0
            y2 = patch_size_h
def cut_cddata(img_A, img_B, img_L, new_path, patch_size_w=256, patch_size_h=256, stride_w=256, stride_h=256):
    imgA = cv2.imread(img_A)
    imgB = cv2.imread(img_B)
    imgL = cv2.imread(img_L)
    newApath = join(new_path,'A')
    newBpath = join(new_path,'B')
    newLpath = join(new_path,'label')
    check_dir(newApath)
    check_dir(newBpath)
    check_dir(newLpath)
    height, width = imgA.shape[:2]
    x1 = 0
    y1 = 0
    x2 = patch_size_w
    y2 = patch_size_h
    n = 0
    while x2 <= width:
        while y2 <= height:
            print("area: ",x1,x2,y1,y2)
            cutAname = 'A_'+str(n)+'.png'
            cutBname = 'B_'+str(n)+'.png'
            cutLname = 'L_'+str(n)+'.png'
            # 保存切割后的图片
            cutA = imgA[y1:y2,x1:x2]
            cutB = imgB[y1:y2,x1:x2]
            cutL = imgL[y1:y2,x1:x2]
            print(cutA.size,' ', cutA.shape)
            cv2.imwrite(join(newApath,cutAname),cutA)
            cv2.imwrite(join(newBpath,cutBname),cutB)
            cv2.imwrite(join(newLpath,cutLname),cutL)
            n=n+1
            # 切割移动方式为从上到下，再从左到右
            y1 = y1 + stride_h
            y2 = y1 + patch_size_h
        # 左右移动
        x1 = x1 + stride_w
        x2 = x1 + patch_size_w
        y1 = 0
        y2 = patch_size_h

if __name__ == '__main__':
    # p_x = range(0, 1024, 256)
    # p_y = range(0, 1024, 256)
    # for i in range(1,17):
    #     bar = i-1
    #     px = p_x[bar//4]
    #     py = p_y[bar%4]
    #     print(px,py)
    # process_data()
    # img_path = '../datasets/BCD/BCDD/label/change_label.tif'
    img_A = '../datasets/BCD/BCDD/A/A.tif'
    img_B = '../datasets/BCD/BCDD/B/B.tif'
    img_L = '../datasets/BCD/BCDD/label/L.tif'
    # img_A = '../datasets/BCD/crop_test/ori/A/0_0.tif'
    # img_B = '../datasets/BCD/crop_test/ori/B/0_0.tif'
    # img_L = '../datasets/BCD/crop_test/ori/label/0_0.tif'
    new_path = '../datasets/BCD/BCDD_cropped256/'
    cut_cddata(img_A, img_B, img_L, new_path)
    # img = cv2.imread(img_path)
    # print(img.shape)
    # img = Image.open(img_path)
    # print(img.size)
    # dir = '../datasets/BCD/BCDD/rename_test'
    # prefix='1'
    # files_rename(dir,prefix)