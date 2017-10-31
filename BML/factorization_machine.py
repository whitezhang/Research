import sys
import numpy as np
from random import normalvariate
import matplotlib.pyplot as plt

class FactorizationMachineClassification():
    def __init__(self):
        pass

    def sigmoid(self, inx):
        return 1.0 / (1 + np.exp(-inx))

    def initialize(self, n, k):
        '''
        '''
        v = np.mat(np.zeros((n, k)))

        for i in xrange(n):
            for j in xrange(k):
                v[i, j] = normalvariate(0, 0.2)
        return v

    def fit(self, dataMatrix, classLabels, k, max_iter, alpha, show_plot=False):
        '''
        train by sgd
        '''
        xmin, xmax = 0, 20
        ymin, ymax = 0, 500
        ys = []
        if show_plot:
            import numpy as np
            plt.axis([xmin, xmax, ymin, ymax])
            #plt.ion()

        m, n = dataMatrix.shape
        w = np.zeros((n, 1))
        w0 = 0
        v = self.initialize(n, k)

        for it in xrange(max_iter):
            for x in xrange(m):
                inter_1 = dataMatrix[x] * v
                inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
                interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
                p = w0 + np.dot(dataMatrix[x], w) + interaction
                loss = self.sigmoid(classLabels[x] * p[0, 0]) - 1

                w0 = w0 - alpha * loss * classLabels[x]
                for i in xrange(n):
                    if dataMatrix[x, i] != 0:
                        w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]

                        for j in xrange(k):
                            v[i, j] = v[i, j] - alpha * loss * classLabels[x] * \
                                    (dataMatrix[x, i] * inter_1[0, j] -\
                                    v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

            if show_plot == True:
                cost = self.get_cost(self.predict(dataMatrix, w0, w, v), classLabels)
                ys.append(cost[0,0])
                ax = plt.gca()
                (xmin_now, xmax_now) = ax.get_xlim()
                if xmax_now > xmax:
                    xmax = x_now + 100
                ymin = min(ys)
                ymax = max(ys)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                plt.scatter(it, cost[0,0])
                plt.pause(0.01)
                print "\t------- iter: ", it, " , cost: ", cost[0,0]
        return w0, w, v

    def get_cost(self, predict, classLabels):
        '''
        '''
        m = len(predict)
        error = 0.0
        for i in xrange(m):
            error -=  np.log(self.sigmoid(predict[i] * classLabels[i] ))
        return error

    def predict(self, dataMatrix, w0, w, v):
        '''
        '''
        m = np.shape(dataMatrix)[0]
        result = []
        for x in xrange(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
             np.multiply(v, v)
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w0 + dataMatrix[x] * w + interaction
            pre = self.sigmoid(p[0, 0])
            result.append(pre)
        return result

    def get_accuracy(self, predict, classLabels):
        '''
        '''
        m = len(predict)
        allItem = 0
        error = 0
        for i in xrange(m):
            allItem += 1
            if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
                error += 1
            elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
                error += 1
            else:
                continue
        return float(error) / allItem

