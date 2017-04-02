import sys
import random
import numpy as np
import argparse

debug = False
model_name = ''

def gen_data(n_fea, n_sample, n_noise):
    basic_fea = np.random.randint(9, size=(1, n_fea+n_noise)) + 1
    noised_fea = np.asmatrix(basic_fea)
    fea = np.asmatrix(basic_fea[:, :n_fea])

    basic_x = np.random.randint(9, size=(n_sample, n_fea+n_noise)) + 1
    noised_x = np.asmatrix(basic_x)
    x = np.asmatrix(basic_x[:, :n_fea])
    noise = np.random.random(size=n_sample)

    y = fea * x.T + noise
    return noised_fea, fea, noised_x, x, y

def print_gen_data(n_W, t_W, n_X, t_X, Y):
    nw_ptr = [str(x) for x in n_W.A.flatten()]
    tw_ptr = [str(x) for x in t_W.A.flatten()]
    y_ptr = [str(x) for x in Y.A.flatten()]

    print '\t\tnW:\t' + '\t'.join(nw_ptr)
    print '\t\ttW:\t' + '\t'.join(tw_ptr)
    print '\t\tY:\t' + '\t'.join(y_ptr)

    """
    print '='*10 + 'noised X' + '='*10
    for line in n_X.A:
        ptr = []
        for g in line:
            ptr.append(str(g))
        print '\t'.join(ptr)

    print '='*10 + 'true X' + '='*10
    for line in t_X.A:
        ptr = []
        for g in line:
            ptr.append(str(g))
        print '\t'.join(ptr)
    """

def succ_ratio(tgt_fea, trained_fea, n_fea):
    abs_tgt_fea = [abs(x) for x in tgt_fea]
    abs_trained_fea = [abs(x) for x in trained_fea]
    L = len(tgt_fea)
    fea_succ_ratio_1 = 0
    fea_succ_ratio_01 = 0
    fea_succ_ratio_001 = 0
    noise_succ_ratio_1 = 0
    noise_succ_ratio_01 = 0
    noise_succ_ratio_001 = 0
    for i in range(L):
        abs_tgt_feai = abs(tgt_fea[i])
        abs_trained_feai = abs(trained_fea[i])
        max_feai = abs_tgt_feai if abs_tgt_feai > abs_trained_feai else abs_trained_feai
        gap_fea = abs(tgt_fea[i] - trained_fea[i]) / max_feai
        if i < n_fea:
            if gap_fea < 0.1:
                fea_succ_ratio_1 += 1
            if gap_fea < 0.01:
                fea_succ_ratio_01 += 1
            if gap_fea < 0.001:
                fea_succ_ratio_001 += 1
        else:
            if gap_fea > 0.9:
                noise_succ_ratio_1 += 1
            if gap_fea > 0.99:
                noise_succ_ratio_01 += 1
            if gap_fea > 0.999:
                noise_succ_ratio_001 += 1
    L = 1. * L
    LF = 1. * n_fea
    LN = 1. * (L - n_fea)
    print '%lf\t%lf\t%lf\t%lf\t%lf\t%lf' % (fea_succ_ratio_1/LF, fea_succ_ratio_01/LF, fea_succ_ratio_001/LF, noise_succ_ratio_1/LN, noise_succ_ratio_01/LN, noise_succ_ratio_001/LN)

def build_and_fit(X, Y):
    if model_name == 1:
        from sklearn import linear_model
        model = linear_model.LinearRegression()
        model.fit(X, Y)
        return model
    elif model_name == 2:
        from sklearn.linear_model import LogistricRegression
        model = LogisticRegression(C=1.0, penalty='l1')
        model.fit(X, Y)
        return model

def train(n_W, t_W, n_X, t_X, Y):
    from sklearn import linear_model
    X = np.asarray(n_X)
    Y = np.asarray(Y.A[0])
    model = build_and_fit(X, Y)

    if debug:
        print '\n\t\tmodel coefs', model.coef_
        print '\t\tmodel intercepts:', model.intercept_
        #print np.dot(n_X, model.coef_.T) + model.intercept_

    succ_ratio(n_W.A.flatten(), model.coef_, t_W.shape[1])
    #model = SelectFromModel(lsvc, prefit=True)
    #X_new = model.transform(X)
    #print X_new

def main():
    argsparser =  argparse.ArgumentParser()
    argsparser.add_argument('-f', '--fea', type=int, help='number of features', default=3)
    argsparser.add_argument('-s', '--sample', type=int, help='number of samples', default=2)
    argsparser.add_argument('-n', '--noise', type=int, help='number of noise', default=1)
    argsparser.add_argument('-d', '--debug', type=bool, help='debug mode', default=False)
    argsparser.add_argument('-fr', '--fea_range', type=str, help='range of features[1-100]')
    argsparser.add_argument('-sr', '--sample_range', type=str, help='range of samples[1-100]')
    argsparser.add_argument('-nr', '--noise_range', type=str, help='range of noise[1-100]')
    argsparser.add_argument('-na', '--model_name', type=int, help='Model\t1.LinearRegression\t2.LogisticRegression', default=1)
    args = argsparser.parse_args()

    global debug, model_name
    range_fea = None
    range_sample = None
    range_noise = None
    model_name = args.model_name

    if args.fea_range != None:
        range_fea = args.fea_range.split('-')
    if args.sample_range != None:
        range_sample = args.sample_range.split('-')
    if args.noise_range != None:
        range_noise = args.noise_range.split('-')
    n_fea = args.fea
    n_sample = args.sample
    n_noise = args.noise
    debug = args.debug

    if range_fea != None:
        s, e = map(int, range_fea)
        for n_fea in range(s, e):
            sys.stdout.write('%d\t%d\t%d\t' % (n_fea, n_sample, n_noise))
            n_W,t_W, n_X, t_X, Y = gen_data(n_fea, n_sample, n_noise)
            train(n_W, t_W, n_X, t_X, Y)
            if debug:
                print_gen_data(n_W, t_W, n_X, t_X, Y)
    elif range_sample != None:
        s, e = map(int, range_sample)
        for n_sample in range(s, e):
            sys.stdout.write('%d\t%d\t%d\t' % (n_fea, n_sample, n_noise))
            n_W,t_W, n_X, t_X, Y = gen_data(n_fea, n_sample, n_noise)
            train(n_W, t_W, n_X, t_X, Y)
            if debug:
                print_gen_data(n_W, t_W, n_X, t_X, Y)
    elif range_noise != None:
        s, e = map(int, range_noise)
        for n_noise in range(s, e):
            sys.stdout.write('%d\t%d\t%d\t' % (n_fea, n_sample, n_noise))
            n_W,t_W, n_X, t_X, Y = gen_data(n_fea, n_sample, n_noise)
            train(n_W, t_W, n_X, t_X, Y)
            if debug:
                print_gen_data(n_W, t_W, n_X, t_X, Y)
    else:
        sys.stdout.write('%d\t%d\t%d\t' % (n_fea, n_sample, n_noise))
        n_W, t_W, n_X, t_X, Y = gen_data(n_fea, n_sample, n_noise)
        train(n_W, t_W, n_X, t_X, Y)
        if debug:
            print_gen_data(n_W, t_W, n_X, t_X, Y)

if __name__ == '__main__':
    main()
