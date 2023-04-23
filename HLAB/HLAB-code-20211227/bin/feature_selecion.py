# encoding:utf-8
import umap
from scipy import stats
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np


def Ttest(X, Y, ttest_num=0):
    Y1 = np.where(Y == 1)[0]
    Y0 = np.where(Y == 0)[0]
    p_list = []
    for i in range(len(X[0])):
        p_l = stats.levene(X[Y1, i], X[Y0, i])[1]
        equal_var = [True if p_l > 0.05 else False]
        p_list.append(stats.ttest_ind(X[Y1, i], X[Y0, i], equal_var=equal_var)[1])

    return p_list, None


def Wtest(X, Y, wtest_num=0):
    Y1 = np.where(Y == 1)[0]
    Y0 = np.where(Y == 0)[0]
    p_list = []
    for i in range(len(X[0])):
        p_list.append(stats.ranksums(X[Y1, i], X[Y0, i])[1])
    return p_list, None


def RF(X, Y, rtest_num=0):
    forest = RandomForestClassifier(random_state=0, n_jobs=1)
    forest.fit(X, Y)
    importance = forest.feature_importances_
    return 1 / (importance + 1e-10), None


def LR_RFE(X, Y, lrtest_num=0):
    clf = LinearRegression()
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X, Y)
    rank = rfe.ranking_
    return rank, None


def SVM_RFE(X, Y, srtest_num=0):
    clf = SVC(kernel='linear', random_state=0)
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X, Y)
    rank = rfe.ranking_
    return rank, None


def init_clf():
    '''
    init classification
    :return: classifiers and their parameters
    '''
    clfs = {'SVM': svm.SVC(probability=True), 'XGBoost': XGBClassifier(probability=True, use_label_encoder=False),
            'KNN': KNeighborsClassifier(), 'NB': GaussianNB(),
            'Bagging': BaggingClassifier(), 'LR': LogisticRegression(),
            'Dtree': DecisionTreeClassifier()}
    return clfs

def visual(data, label, x=0, y=1):
    import matplotlib.pyplot as plt
    pos = data[np.where(label == 1)[0]][:, :6]  # (n,2)
    neg = data[np.where(label == 0)[0]][:, :6]
    plt.scatter(pos[:, x], pos[:, y], c='r', marker='x')
    plt.scatter(neg[:, x], neg[:, y], c='b', marker='.')
    plt.show()


def fs(train, val, test, params_list):
    """
    [train_feature, train_label], [eval_feature, eval_label], [test_feature, test_label]
    :param train: [train_feature, train_label]
    :param val: [eval_feature, eval_label]
    :param test:  [test_feature, test_label]
    :return:
    """
    fs_list = {
        'RF': RF,
        'Ttest': Ttest,
        'Wtest': Wtest,
        'SVM_RFE': SVM_RFE,
        'LR_RFE': LR_RFE,
    }
    # After the pipeline performs feature processing and model training, the result is directly a trained classifier
    clf_name = params_list[0]
    fs_name = fs_list[params_list[1]]
    umap_params = int(params_list[2])
    fs_params = params_list[3]

    clfs = init_clf()
    sd = StandardScaler()
    train[0] = sd.fit_transform(train[0], train[1])
    val[0] = sd.transform(val[0])
    test[0] = sd.transform(test[0])

    #  Feature dimensionality reduction
    select = umap.UMAP(n_components=umap_params)
    select.fit(train[0], train[1])
    X_train = select.transform(train[0])
    X_val = select.transform(val[0])
    X_test = select.transform(test[0])

    #  Feature selection
    skb = SelectKBest(score_func=fs_name, k=int(X_train.shape[1]))
    skb.fit(X_train, train[1])
    skb.k = int(X_train.shape[1] * fs_params)
    X_train_i = skb.transform(X_train)
    X_val_i = skb.transform(X_val)
    X_test_i = skb.transform(X_test)

    clf = clfs[clf_name]
    clf.fit(X_train_i, train[1])
    scorce = clf.score(X_val_i, val[1])

    y_pres = clf.predict(X_test_i)  # predict results


    return y_pres


def reshape(a):
    a[0].shape = (len(a[1]), -1)
    a[1].shape = (-1,)

def metrics(preds, labels, probs):
    acc = (preds == labels).mean()
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs)
    return {
        "acc": acc,
        "mcc": mcc,
        "auc": auc,
        "sn": sn,
        "sp": sp,
    }


