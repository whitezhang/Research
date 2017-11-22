import sys
import numpy as np
import json

from chimerge import ChiMerge
from chi2 import Chi2
import utils

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def sparse_to_matrix(data):
    n = len(data)
    vals = list(set([x for g in data for x in g]))
    fea2idx = {v: vals.index(v) for v in vals}
    m = len(vals)
    X = np.zeros((n, m))
    for i in range(n):
        g = data[i]
        for v in g:
            idx = fea2idx[v]
            X[i, idx] = 1
    return X


def output_case(data, feature_names, label):
    data = np.asarray(data).flatten()
    print data
    L = len(feature_names)
    ptr = {}
    for i in range(L):
        ptr[feature_names[i]] = data[i]
    for k, v in ptr.items():
        if v != 0:
            print str(k) + ':' + str(v),
    print label


def feature_importance_learning(dataTrain, labelTrain, feature_names, cut_ratio):
    '''
    LR, RF
    '''
    n_folds = 3

    feature_candidates = {}
    features_learned = {}
    depth = 40
    for i, (train_index, test_index) in enumerate(StratifiedKFold(np.asarray(labelTrain).flatten(), n_folds=n_folds, shuffle=True)):
        X_train, X_test = dataTrain[train_index], dataTrain[test_index]
        y_train, y_test = labelTrain[train_index], labelTrain[test_index]
        clf = RandomForestClassifier(max_depth=depth, random_state=0)
        clf.fit(X_train, y_train)
        for i in range(len(feature_names)):
            features_learned[feature_names[i]] = clf.feature_importances_[i]

    feature_num = len(feature_names)
    cut_number = int(feature_num * cut_ratio)
    feature_candidates = utils.sortDictByValue(features_learned, True)
    print 'Features numbers: ', feature_num, 'Now: ', cut_number

    features = [x[0] for x in feature_candidates[:cut_number]]
    return features




def process_adult_trad(attribute_column, min_expected_value, max_number_intervals, threshold, debug_info):
    attributes = [('age', 'i8'), ('workclass', 'S40'), ('fnlwgt', 'i8'), ('education', 'S40'), ('education-num', 'i8'), ('marital-status', 'S40'), ('occupation', 'S40'), ('relationship', 'S40'), ('race', 'S40'), ('sex', 'S40'), ('capital-gain', 'i8'), ('capital-loss', 'i8'), ('hours-per-week', 'i8'), ('native-country', 'S40'), ('pay', 'S40')]
    datatype = np.dtype(attributes)
    # BOW model
    data, Y, feature_names = _readAdultDataSet(attribute_column, attributes)

    n_folds = 3
    #dataTrain = np.asarray(data)
    #labelTrain = np.asarray(Y)

    #for cut_ratio in [0.2, 0.4, 0.6, 0.8, 1]:
    for cut_ratio in [1]:
        feature_selected = feature_importance_learning(np.asarray(data), np.asarray(Y), feature_names, cut_ratio)
        data_idx = []
        for i in range(len(feature_names)):
            if feature_names[i] in feature_selected:
                data_idx.append(i)
        data_selected = data[:, data_idx]
        dataTrain = np.asarray(data_selected)
        labelTrain = np.asarray(Y)

        alphas = [0.5, 1, 5, 10, 100]
        #alphas = [10, 20, 50, 200]
        for alpha in alphas:
            score_train = []
            score_test = []
            for i, (train_index, test_index) in enumerate(StratifiedKFold(np.asarray(Y).flatten(), n_folds=n_folds, shuffle=True)):
                #clf = SVC(class_weight='balanced', kernel='linear', C=alpha)
                #clf = RandomForestClassifier(max_depth=alpha, random_state=0)
                clf = LogisticRegression(penalty='l1', C=alpha)
                #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 5), random_state=1)
                X_train, X_test = dataTrain[train_index], dataTrain[test_index]
                y_train, y_test = labelTrain[train_index], labelTrain[test_index]
                clf.fit(X_train, y_train)
                pred_train = clf.predict(X_train)
                pred_test = clf.predict(X_test)
                score_train.append(metrics.accuracy_score(y_train, pred_train))
                score_test.append(metrics.accuracy_score(y_test, pred_test))

                """
                class_coefs1 = {}
                print clf.coef_.shape
                for i in range(clf.coef_.shape[1]):
                    class_coefs1[feature_names[i]] = clf.coef_[0, i]
                sorted_class_coefs1 = utils.sortDictByValue(class_coefs1, True)
                print sorted_class_coefs1
                """
                """
                for i in range(len(y_train)):
                    if y_train[i] != pred_train[i]:
                        output_case(data[i,:], feature_names, y_train[i])
                """
            print 'cut_ratio:', cut_ratio, 'alpha:', alpha, 'Average accuracy, train: ', 1.*sum(score_train)/len(score_train), 'test: ', 1.*sum(score_test)/len(score_test)


def process_adult_bml(attribute_column, min_expected_value, max_number_intervals, threshold, debug_info):
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

    """
    dv = DictVectorizer(sparse=False)
    dataTrain = dv.fit_transform(X_parsed)
    labelTrain = Y[:,0]
    """

    # fm training
    #print af_model.reversed_table
    ori_features_len = len(set(af_model.reversed_table.keys()))
    parsed_features_len = len(set(af_model.reversed_table.values()))
    print "Features space transforms from %d-dim to %d-dim" % (ori_features_len, parsed_features_len)
    dataTrain = sparse_to_matrix(X_parsed)
    labelTrain = Y[:,0]
    """
    for i in range(dataTrain.shape[0]):
        for j in range(dataTrain.shape[1]):
            print dataTrain[i,j],
        print ''
    """

    # kfold validation
    from factorization_machine import FactorizationMachineClassification
    from sklearn.svm import SVC
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import StratifiedKFold

    n_folds = 3

    # This following models contains Sklearn models and FM model, pick one
    # SK models
    alphas = [0.5, 1, 5, 10, 100, 1000]
    for alpha in alphas:
        score_train = []
        score_test = []
        for i, (train_index, test_index) in enumerate(StratifiedKFold(np.asarray(labelTrain).flatten(), n_folds=n_folds, shuffle=True)):
            #clf = SVC(class_weight='balanced', kernel='linear', C=alpha)
            clf = LogisticRegression(penalty='l2', C=alpha)
            #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 3), random_state=1, max_iter=1000)
            X_train, X_test = dataTrain[train_index], dataTrain[test_index]
            y_train, y_test = labelTrain[train_index], labelTrain[test_index]
            clf.fit(X_train, y_train)
            pred_train = clf.predict(X_train)
            pred_test = clf.predict(X_test)
            score_train.append(metrics.accuracy_score(y_train, pred_train))
            score_test.append(metrics.accuracy_score(y_test, pred_test))
        print 'alpha:', alpha, 'Average accuracy, train: ', 1.*sum(score_train)/len(score_train), 'test: ', 1.*sum(score_test)/len(score_test)
    return

    # FM model
    fm = FactorizationMachineClassification()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(np.asarray(labelTrain).flatten(), n_folds=n_folds, shuffle=True)):
        X_train, X_test = dataTrain[train_index], dataTrain[test_index]
        y_train, y_test = labelTrain[train_index], labelTrain[test_index]
        w0, w, v = fm.fit_and_validate(np.mat(X_train), y_train, np.mat(X_test), y_test, 3, 10000, 0.01, True)
        break


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
    pathfn = 'adult/adult.small'
    #pathfn = 'adult/adult.1w'
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

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(data)
    return np.matrix(X, dtype='i8'), np.matrix(Y).T, dv.get_feature_names()


# ChiMerge paper: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf
if __name__ == '__main__':
    process_adult_bml(attribute_column=-1, min_expected_value=0.5, max_number_intervals=15, threshold=4.61, debug_info=False)
    #process_adult_trad(attribute_column=-1, min_expected_value=0.5, max_number_intervals=6, threshold=4.61, debug_info=False)
