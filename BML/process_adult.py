import sys
import numpy as np
import json

from chimerge import ChiMerge
from chi2 import Chi2
import utils

def sparse_to_matrix(data):
    n = len(data)
    vals = list(set([x for g in data for x in g]))
    fea2idx = {v: vals.index(v) for v in vals}
    m = len(vals)
    X = np.zeros((n, m))
    print 'feature size', m
    for i in range(n):
        g = data[i]
        for v in g:
            idx = fea2idx[v]
            X[i, idx] = 1
    return X

def process_adult(attribute_column, min_expected_value, max_number_intervals, threshold, debug_info):
    attributes = [('age', 'i8'), ('workclass', 'S40'), ('fnlwgt', 'i8'), ('education', 'S40'), ('education-num', 'i8'), ('marital-status', 'S40'), ('occupation', 'S40'), ('relationship', 'S40'), ('race', 'S40'), ('sex', 'S40'), ('capital-gain', 'i8'), ('capital-loss', 'i8'), ('hours-per-week', 'i8'), ('native-country', 'S40'), ('pay', 'S40')]
    datatype = np.dtype(attributes)

    chi = ChiMerge(min_expected_value, max_number_intervals, threshold, debug_info)
    # BOW model
    data, Y, feature_names = _readAdultDataSet(attribute_column, attributes)

    # Chimerge
    discretizationIntervals = {}
    discretizationDtype = []
    for i in range(data.shape[1]):
        chiData = np.concatenate((data[:,i], Y), axis=1)
        chi.loadData(chiData, False)
        chi.generateFrequencyMatrix()
        chi.chimerge()
        #chi.printDiscretizationInfo(feature_names[i])
        discretizationIntervals[feature_names[i]] = chi.frequency_matrix_intervals
        discretizationDtype.append((feature_names[i], 'i8'))

    # addfeatures
    from addfeatures import AddFeatures
    af_model = AddFeatures()
    X_parsed = []
    for i in range(data.shape[0]):
        input_stream = np.zeros((1,),dtype=object)
        input_stream[0] = np.asarray(data[i,:])
        X_slots = af_model.fit_transform(data=input_stream, dttyp=np.dtype(discretizationDtype), discret_intervals=discretizationIntervals)
        X_parsed.append(X_slots)

    # fm training
    dataTrain = sparse_to_matrix(X_parsed)
    labelTrain = Y[:,0]

    from factorization_machine import FactorizationMachineClassification
    fm = FactorizationMachineClassification()
    w0, w, v = fm.fit(np.mat(dataTrain), labelTrain, 3, 10000, 0.01, True)
    pred_result = fm.predict(np.mat(dataTrain), w0, w, v)
    print 1 - fm.get_accuracy(pred_result, labelTrain)


def _readAdultDataSet(attribute_column=-1, attributes=None):
    """
    Reference: http://archive.ics.uci.edu/ml/machine-learning-databases/adult/
    e.g. 39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K

    >50K, <=50K.

    age: continuous.
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    :return:
    """
    if attribute_column < -1 or attribute_column > 15:
        return
    if attribute_column== -1:
        attribute_columns = range(15)
    else:
        attribute_columns = [attribute_column]
    if attributes == None:
        return
    datatype = np.dtype(attributes)

    #pathfn = 'adult/adult.data'
    pathfn = 'adult/adult.small200'
    data = []
    Y = []

    with open(pathfn, 'r') as f:
        for line in f:
            tmpdict = {}
            tmp = line.replace(' ', '').replace(':', '-').strip().split(',')
            tmp = np.array(tuple(tmp), dtype=datatype)
            for g in attributes:
                typ = g[0]
                value = tmp[typ]

                if g[0] == 'pay' and value == '>50K':
                    Y.append(1)
                elif g[0] == 'pay'  and value == '<=50K':
                    Y.append(-1)
                elif value.dtype == np.dtype('S40'):
                    #tag = str(typ) + BaseC.DISCRET_DELIMITER + str(value)
                    tag = utils.mergeKeyValue(str(typ), str(value), 'discret')
                    tmpdict[tag] = 1
                else:
                    tmpdict[typ] = value
            data.append(tmpdict)

    from sklearn.feature_extraction import DictVectorizer
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(data)
    return np.matrix(X, dtype='i8'), np.matrix(Y).T, dv.get_feature_names()


# ChiMerge paper: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf
if __name__ == '__main__':
    process_adult(attribute_column=-1, min_expected_value=0.5, max_number_intervals=6, threshold=4.61, debug_info=False)
