import numpy as np
import scipy.sparse as sp
import logging
import sys


def calcos(wf1, wf2):
    sim = 0
    norm = 0
    n2 = 0
    for key, value in wf1.items():
        # print key
        if key in wf2:
            sim += wf1[key]*wf2[key]
            norm += wf2[key]**2
            n2 += wf1[key]**2
    # return 1.0*sim, labelwf['label']
    # return 1.0*sim/math.sqrt(norm)/math.sqrt(n2)/200, labelwf['label']
    if norm == 0:
        return -1, ""
    return 1.0*sim/math.sqrt(norm)

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

def str_to_wf(wf_list, convertf=convertf_100, normed=True, topk=30):
    r = dict()
    norm = 0.0

    wfs = [p.split(';') for p in wf_list.split('|')]
    for wf in wfs[:topk]:
        try:
            if len(wf) <= 1:
                pass
            if len(wf) == 2:
                k = wf[0].replace(':', '')
                if k == 0:
                    continue
                v = np.float(100 - int(wf[1]))
                #v = abs(int(wf[1]))
                v = np.float(convertf(v))
                r[k] = v
                norm += v * v
            elif len(wf) == 3:
                #k = long(wf[0], base=16)
                #k = wf[0]
                k = wf[0].replace(':', '')
                if k == 0:
                    continue
                v = np.float(100 - int(wf[2]))
                #v = abs(int(wf[2]))
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

