# from turtle import color
from cProfile import label
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
methods = ['FC-EF','FC-Siam-diff','FC-Siam-conv','DSIFNet','DSAMNet','SNUNet','FCCDN','ChangeFormer','DAHRNet']

macs = [3.562,4.699,5.303,82.264,75.29,123.12,12.522,202.829,13.157]

params = [1.348,1.347,1.543,35.728,16.951,27.068,6.307,41.005,14.656]

F1 = [0.7722,0.7914,0.7905,0.8055,0.7855,0.8185,0.8075,0.8071,0.8336]
colors = ['b','c','g','k','m','r','y']
markers = [r"$\circledast$",'v','.','s','+','h','x',r"$\diamond$",r"$\bigstar$",'^']
def plot_macs_params_f1():
    fig = plt.figure(figsize=(8,4))
    scale=50
    with plt.style.context(['ieee', 'grid', 'scatter']):
        ax = plt.subplot(121)
        for i in range(len(methods)):
            ax.scatter(macs[i],F1[i],s=scale,c=colors[(i+4)%7], marker=markers[i], label=methods[i])
        ax.legend(title='Methods',loc='lower right')
        
        ax.set_xlabel('MACs')
        ax.set_ylabel('F1_1')
        ax.set_ylim(bottom=0.5,top=0.85)
        # fig.savefig('macs_f1.pdf')
        ax = plt.subplot(122)
        for i in range(len(methods)):
            ax.scatter(params[i],F1[i],s=scale,c=colors[(i+4)%7], marker=markers[i], label=methods[i])
        ax.legend(title='Methods',loc='lower right')
        
        ax.set_xlabel('PARAMs')
        # ax.set_ylabel('F1_1')
        ax.set_ylim(bottom=0.5,top=0.85)
        plt.show()
        fig.savefig('p.jpg')
        fig.savefig('macs_params_f1.pdf')
def plot_bs_f1():
    f1 = [83.46,83.49,83.36,83.31]
    bs = [4,8,16,32]
    pparam = dict(xlabel='Batch size', ylabel=r'F1 (%)')
    fig = plt.figure()
    with plt.style.context(['ieee']):
        
        f = plt.figure(figsize=(7, 4))
        plt.plot(bs, f1, label='Twitter', marker='o', color='black')
        for x, y in zip(bs, f1):
            plt.text(x-1.12, y+0.03, '%.2f' % y,fontdict={'fontsize':8})
        plt.xlim(left=1, right=40)#x 轴范围
        plt.ylim(bottom=83, top=84)# y 轴范围
        plt.xticks(bs, ['4', '8', '16', '32'], fontsize=12)# 把x 的坐标刻度用第二个参数的字符串代替
        plt.ylabel('F1 (%)', fontsize=14)
        plt.xlabel('Batch size', fontsize=14)
        plt.grid(alpha=0.5, linestyle='--')# 设置网络线
        plt.show()
        f.savefig("bs_f1.pdf", bbox_inches='tight')
        f.savefig("bs_f1.png", bbox_inches='tight')

def plot_lr_f1():
    f1 = [82.74,83.15,83.36,83.28]# 顺序 '1e-6', '1e-5', '1e-4', '1e-3'
    lr = [2, 4, 8, 16]# 顺序 '1e-6', '1e-5', '1e-4', '1e-3'
    pparam = dict(xlabel='Batch size', ylabel=r'F1 (%)')
    fig = plt.figure()
    with plt.style.context(['ieee']):
        
        f = plt.figure(figsize=(7, 4))
        plt.plot(lr, f1, label='Twitter', marker='o', color='black')
        for x, y in zip(lr, f1):
            py = y+0.03 if x !=2 else y-0.05
            plt.text(x-0.57, py, '%.2f' % y,fontdict={'fontsize':8})
        plt.xlim(left=0, right=20)#x 轴范围
        plt.ylim(bottom=82.5, top=83.5)# y 轴范围
        plt.xticks(lr, [r"$1\times10^{-6}$", r'$1\times10^{-5}$', r'$1\times10^{-4}$', r'$1\times10^{-3}$'], fontsize=8)# 把x 的坐标刻度用第二个参数的字符串代替
        plt.ylabel('F1 (%)', fontsize=14)
        plt.xlabel('Learning rate', fontsize=14)
        plt.grid(alpha=0.5, linestyle='--')# 设置网络线
        plt.show()
        f.savefig("lr_f1.pdf", bbox_inches='tight')
        f.savefig("lr_f1.png", bbox_inches='tight')
if __name__ == '__main__':
    plot_lr_f1()
