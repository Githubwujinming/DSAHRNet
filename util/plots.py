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

if __name__ == '__main__':
    plot_macs_params_f1()
