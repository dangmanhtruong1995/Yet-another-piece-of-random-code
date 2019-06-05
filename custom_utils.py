import numpy as np
import scipy.io as spio
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

def calculatePRF(confmat):
    M = confmat.shape[0]
    P = np.zeros(M)
    R = np.zeros(M)
    F1 = np.zeros(M)
    col_list = np.sum(confmat, 0)
    row_list = np.sum(confmat, 1)
    for i in range(M):
        col = col_list[i]
        if col != 0:
            P[i] = confmat[i][i] / (col * 1.0)

        row = row_list[i]
        if row != 0:
            R[i] = confmat[i][i] / (row * 1.0)

        total = P[i] + R[i]
        if total != 0:
            F1[i] = (2 * P[i] * R[i]) / (total * 1.0)

    PAvg = np.mean(P)
    RAvg = np.mean(R)
    F1Avg = np.mean(F1)
    result = {}
    result['precision'] = PAvg
    result['recall'] = RAvg
    result['f1'] = F1Avg
    return result

def load_cv_mat(file_name):
    mat = spio.loadmat(file_name, squeeze_me=True)
    cv = mat['cv']
    return cv

def load_data(file_name):
    # If only there is a Python equivalent of Matlab's importdata ...
    try:
        data = np.loadtxt(file_name, delimiter = ',')
    except:
        data = np.loadtxt(file_name)
    return data

def create_metadata(X, Y, clfs_list, num_of_classes, num_of_cv_folds):
    """
    Create metadata

    Input:

    - X: input instances

    - Y: labels

    - clfs_list: List of classifiers (e.g: clf = MultinomialNB())

    - num_of_classes: Number of classifiers

    - num_of_cv_folds: Number of cross-validation num_of_cv_folds

    Output:

    - metadata: Metadata created from clfs_list

    """
    num_of_clfs = len(clfs_list)
    num_of_instances = X.shape[0]
    metadata = np.zeros((num_of_instances, num_of_clfs * num_of_classes))

    kf = KFold(n_splits = num_of_cv_folds)
    kf_split = kf.split(X)
    for train_idx, test_idx in kf_split:
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        clfs_idx = 0
        for clf in clfs_list:
            model = clf.fit(X[train_idx, :], Y[train_idx])
            prob = model.predict_proba(X[test_idx, :])
            metadata[test_idx, clfs_idx: clfs_idx + num_of_classes] = prob
            clfs_idx += num_of_classes

    return metadata

def create_metadata_for_test(X, num_of_classes, model_list):
    num_of_clfs = len(model_list)
    num_of_instances = X.shape[0]

    metadata = np.zeros((num_of_instances, num_of_clfs * num_of_classes))
    clfs_idx = 0
    for model in model_list:
        prob = model.predict_proba(X)
        metadata[:, clfs_idx: clfs_idx + num_of_classes] = prob
        clfs_idx += num_of_classes 
    return metadata

def collect_result(Y_test, Y_pred):
    confmat = confusion_matrix(Y_test, Y_pred)
    result = calculatePRF(confmat)
    result['err'] = 1 - (np.sum(np.diag(confmat)) / (1.0 * np.sum(confmat)))
    return result
