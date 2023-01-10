# coding=utf-8
from crop_img import check_dir
import os, random, shutil
# random.seed(0)
# 将图片拆分成训练集train(0.8)和验证集val(0.2)

def moveFile(Dir,splited_dir,train_ratio=0.8,val_ratio=0.1, test_ratio=0.1):

    if not os.path.exists(os.path.join(splited_dir, 'train')):
        os.makedirs(os.path.join(splited_dir, 'train'))
    
    if not os.path.exists(os.path.join(splited_dir, 'val')):
        os.makedirs(os.path.join(splited_dir, 'val'))
    if not os.path.exists(os.path.join(splited_dir, 'test')):
        os.makedirs(os.path.join(splited_dir, 'test'))

    filenames = []
    Adir = os.path.join(Dir,'A')
    Bdir = os.path.join(Dir,'B')
    Ldir = os.path.join(Dir,'label')
    for root,dirs,files in os.walk(Adir):
        for name in files:
            filenames.append(name)
        break
    
    filenum = len(filenames)
    # print(filenames)
    num_train = int(filenum * train_ratio)
    num_val = int(filenum * val_ratio)
    sample_train = random.sample(filenames, num_train)

    for name in sample_train:
        check_dir(os.path.join(splited_dir, 'train','A'))
        check_dir(os.path.join(splited_dir, 'train','B'))
        check_dir(os.path.join(splited_dir, 'train','label'))
        shutil.copy(os.path.join(Adir, name), os.path.join(splited_dir, 'train','A',name))
        shutil.copy(os.path.join(Bdir, name.replace('A_','B_')), os.path.join(splited_dir, 'train','B', name.replace('A_','L_')))
        shutil.copy(os.path.join(Ldir, name.replace('A_','L_')), os.path.join(splited_dir, 'train','label', name.replace('A_','L_')))

    sample_val_test = list(set(filenames).difference(set(sample_train)))
    sample_val = random.sample(sample_val_test, num_val)

    for name in sample_val:
        check_dir(os.path.join(splited_dir, 'val','A'))
        check_dir(os.path.join(splited_dir, 'val','B'))
        check_dir(os.path.join(splited_dir, 'val','label'))
        shutil.copy(os.path.join(Adir, name), os.path.join(splited_dir, 'val','A',name))
        shutil.copy(os.path.join(Bdir, name.replace('A_','B_')), os.path.join(splited_dir, 'val','B', name.replace('A_','L_')))
        shutil.copy(os.path.join(Ldir, name.replace('A_','L_')), os.path.join(splited_dir, 'val','label', name.replace('A_','L_')))
    sample_test = list(set(sample_val_test).difference(set(sample_val)))

    for name in sample_test:

        check_dir(os.path.join(splited_dir, 'test','A'))
        check_dir(os.path.join(splited_dir, 'test','B'))
        check_dir(os.path.join(splited_dir, 'test','label'))
        shutil.copy(os.path.join(Adir, name), os.path.join(splited_dir, 'test','A',name))
        shutil.copy(os.path.join(Bdir, name.replace('A_','B_')), os.path.join(splited_dir, 'test','B', name.replace('A_','L_')))
        shutil.copy(os.path.join(Ldir, name.replace('A_','L_')), os.path.join(splited_dir, 'test','label', name.replace('A_','L_')))

    # for name in sample_val:
    #     shutil.move(os.path.join(Dir, name), os.path.join(Dir, 'val'))



def splitData(Dir,splited_dir,train_ratio=10):
    if not os.path.exists(os.path.join(splited_dir, 'train%s'%train_ratio)):
        os.makedirs(os.path.join(splited_dir, 'train%s'%train_ratio))

    filenames = []
    Adir = os.path.join(Dir,'A')
    Bdir = os.path.join(Dir,'B')
    Ldir = os.path.join(Dir,'label')
    for root,dirs,files in os.walk(Adir):
        for name in files:
            filenames.append(name) 
        break
    filenum = len(filenames)
    # print(filenames)
    num_train = int(filenum * train_ratio/100)
    sample_train = random.sample(filenames, num_train)
    for name in sample_train:
        check_dir(os.path.join(splited_dir, 'train%s'%train_ratio,'A'))
        check_dir(os.path.join(splited_dir, 'train%s'%train_ratio,'B'))
        check_dir(os.path.join(splited_dir, 'train%s'%train_ratio,'label'))
        shutil.copy(os.path.join(Adir, name), os.path.join(splited_dir, 'train%s'%train_ratio,'A',name))
        shutil.copy(os.path.join(Bdir, name), os.path.join(splited_dir, 'train%s'%train_ratio,'B', name))
        shutil.copy(os.path.join(Ldir, name), os.path.join(splited_dir, 'train%s'%train_ratio,'label', name))

if __name__ == '__main__':
    # Dir = '../datasets/BCD/LEVIR-CD_cropped256/'
    # splited_dir = '../datasets/BCD/BCDD_splited_cropped256/'
    # check_dir(splited_dir)
    # moveFile(Dir,splited_dir)
    Dir = '../datasets/SYSU-CD/train'
    splited_dir = '../datasets/SYSU-CD'
    splitData(Dir, splited_dir, train_ratio=10)
    # for root,dirs,files in os.walk(Dir):
    #     for name in dirs:
    #         folder = os.path.join(root, name)
    #         print("正在处理:" + folder)
    #         # moveFile(folder)
    #     print("处理完成")
    #     break

