import sys
import numpy as np

class BaseModel():
    def __init__(self):
        pass

    def fit(self, dataMatrix, classLabels, k, max_iter, alpha):
        '''
        train by sgd
        '''
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

            if it % 10 == 0:
                print "\t------- iter: ", it, " , cost: ", \
                self.get_cost(self.predict(dataMatrix, w0, w, v), classLabels)
        return w0, w, v

