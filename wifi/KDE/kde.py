# May be not need
import sys
import random
import numpy as np
import argparse
from sklearn.neighbors.kde import KernelDensity

debug = False
VALID_KERNELS = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

def build_and_plot(kernel_name, X):
    kde = KernelDensity(kernel=kernel_name, bandwidth=0.2).fit(X)

def kde(kernel_name, s_pic, e_pic):
    wfs_map = {}
    for line in sys.stdin:
        wfs = line.strip('\n').split('|')
        for wf in wfs:
            wf = wf.split(';')
            if len(wf) != 3:
                continue
            ap = wf[0]
            sig = int(wf[2])
            if ap not in wfs_map:
                wfs_map[ap] = [sig]
            else:
                wfs_map[ap].append(sig)

    sorted_wfs_map = [ (k, wfs_map[k]) for k in sorted(wfs_map.keys()) ]
    build_and_plot(kernel_name, sorted_wfs_map[s_pic: e_pic)

def main():
    argsparser =  argparse.ArgumentParser()
    argsparser.add_argument('-r', '--range', type=str, help='Range of pictures', default='1-4')
    argsparser.add_argument('-k', '--kernel_name', type=int, help='Kernel: gaussian, tophat, epanechnikov, exponential, linear, cosine', default=1)
    args = argsparser.parse_args()

    global debug
    kernel_name = args.kernel_name
    s_pic, e_pic = map(int, args.range.split('-'))

    kde(kernel_name, s_pic, e_pic+1)

if __name__ == '__main__':
    main()
