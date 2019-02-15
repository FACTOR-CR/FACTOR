'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import scipy as sp
from functools import reduce


def binarize_predictions(array, task='binary.classification'):
    ''' Turn predictions into decisions {0,1} by selecting the class with largest 
    score for multiclass problems and thresholding at 0.5 for other cases.'''
    # add a very small random value as tie breaker (a bit bad because this changes the score every time)
    # so to make sure we get the same result every time, we seed it    
    # eps = 1e-15
    # np.random.seed(sum(array.shape))
    # array = array + eps*np.random.rand(array.shape[0],array.shape[1])
    bin_array = np.zeros(array.shape)
    if (task != 'multiclass.classification') or (array.shape[1] == 1):
        bin_array[array >= 0.5] = 1
    else:
        sample_num = array.shape[0]
        for i in range(sample_num):
            j = np.argmax(array[i, :])
            bin_array[i, j] = 1
    return bin_array

def acc_stat(solution, prediction):
    ''' Return accuracy statistics TN, FP, TP, FN
     Assumes that solution and prediction are binary 0/1 vectors.'''
    # This uses floats so the results are floats
    TN = sum(np.multiply((1 - solution), (1 - prediction)))
    FN = sum(np.multiply(solution, (1 - prediction)))
    TP = sum(np.multiply(solution, prediction))
    FP = sum(np.multiply((1 - solution), prediction))
    # print "TN =",TN
    # print "FP =",FP
    # print "TP =",TP
    # print "FN =",FN
    return (TN, FP, TP, FN)

def mvmean(R, axis=0):
    ''' Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten.'''
    if len(R.shape) == 0: return R
    average = lambda x: reduce(lambda i, j: (0, (j[0] / (j[0] + 1.)) * i[1] + (1. / (j[0] + 1)) * j[1]), enumerate(x))[
        1]
    R = np.array(R)
    if len(R.shape) == 1: return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))

def bac_metric(solution, prediction, task='binary.classification'):
    ''' Compute the normalized balanced accuracy. The binarization and 
    the normalization differ for the multi-label and multi-class case. '''
    label_num = solution.shape[1]
    score = np.zeros(label_num)
    bin_prediction = binarize_predictions(prediction, task)
    [tn, fp, tp, fn] = acc_stat(solution, bin_prediction)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum(eps, tp)
    pos_num = sp.maximum(eps, tp + fn)
    tpr = tp / pos_num  # true positive rate (sensitivity)
    if (task != 'multiclass.classification') or (label_num == 1):
        tn = sp.maximum(eps, tn)
        neg_num = sp.maximum(eps, tn + fp)
        tnr = tn / neg_num  # true negative rate (specificity)
        bac = 0.5 * (tpr + tnr)
        base_bac = 0.5  # random predictions for binary case
    else:
        bac = tpr
        base_bac = 1. / label_num  # random predictions for multiclass case
    bac = mvmean(bac)  # average over all classes
    # Normalize: 0 for random, 1 for perfect
    score = (bac - base_bac) / sp.maximum(eps, (1 - base_bac))
    return score

def bac_multiclass(solution, prediction):
    prediction = prediction.reshape((prediction.shape[0],1))
    return bac_metric(solution, prediction, task='multiclass.classification')

from sklearn import metrics

def precision_score(solution, prediction):
    prediction = prediction.reshape((prediction.shape[0],1))
    return metrics.precision_score(solution, prediction, average='macro')
