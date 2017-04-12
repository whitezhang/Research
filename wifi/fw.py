import numpy as np
import scipy.sparse as sp
import logging
import sys

def get_data(lines, sep='\t'):
    for line in lines:
        line = line.strip('\n').split(sep)
        yield line

def convertf_100(i):
    return 100 - int(i)

def convertf_1(i):
    return 1

def wf_to_str(wf_list):
    pass

def str_to_wf(wf_list, convertf=convertf_100, normed=True):
    r = dict()
    norm = 0.0

    wfs = [p.split(';') for p in wf_list.split('|')]
    for wf in wfs:
        try:
            if len(wf) < 2 or len(wf[1]) == 0:
                continue
            #k = long(wf[0], base=16)
            k = wf[0]
            if k == 0:
                continue
            #v = np.float(100 - int(wf[1]))
            v = abs(int(wf[2]))
            v = np.float(convertf(v))
            r[k] = v
            norm += v * v
        except:
            t,v = sys.exc_info()[:2]
            sys.stderr.write(wf_list + str(t) + str(v) + '\n')
            #logging.warning(str(t)+str(v) + ':' + wf_list)
            continue
    if normed and norm > 0:
        norm = np.sqrt(norm)
        for k in r:
            r[k] /= norm
    return r

