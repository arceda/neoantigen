# coding=utf-8
import pandas as pd
from HLAB.bin import feature_extract as fe
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import numpy as np
import os

currentPath = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(currentPath)


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

def querySupportAllel():
    df = pd.read_excel(os.path.join(parent_path, 'source/allel.xlsx'))
    allel_list = df['ALLEL'].tolist()
    return allel_list

def queryPeptideLengthByAllel(allel):
    df = pd.read_excel(os.path.join(parent_path, 'source/allel.xlsx'))
    support_length = df[df.ALLEL == allel]['peptide_Length'].values
    if support_length.size > 0:
        support_length = support_length.tolist()[0].split(',')
        support_length = list(map(int, support_length))
    return support_length

def predict(allel, peptide, peptide_length):
    # Determine whether the input peptide sequence and length are consistent.

    try:
        peptide_length = int(peptide_length)
    except ValueError:
        return ('Your input does not conform to the specification, \n'
                'please type <help predict> to understand the standard input format of the predict function')
    if len(peptide) != peptide_length:
        error_message = 'The input peptide sequence and length are inconsistent.\n' \
                        'Please check your input data.'
        return error_message
    
    # Determine whether HLAB supports prediction of allel and peptide length.
    support_length = queryPeptideLengthByAllel(allel)
    if peptide_length not in support_length:
        error_message = 'HLAB does not support prediction of input allel and peptide length.\n' \
                        'Please use the query function to query the allele and length predicted by HLAB.'
        return error_message
    
    #Extract features
    model_path = os.path.join(os.path.dirname(parent_path), 'HLAB-code-20211227')
    model_path = os.path.join(model_path, 'rnnmodels')
    bert_feature = fe.Feature_Extract(allel, peptide, model_path).embedding()
    bert_feature = bert_feature.values
    hla = allel.replace(':', '_').replace('*', '_')
    try:
        sd_model_name = hla + '_sd_model.pkl'
        sd_model = joblib.load(os.path.join(parent_path, 'model\%s\%s' % (peptide_length, sd_model_name)))
        select_model_name = hla + '_select_model.pkl'
        select_model = joblib.load(os.path.join(parent_path, 'model\%s\%s' % (peptide_length, select_model_name)))
        skb_model_name = hla + '_skb_model.pkl'
        skb_model = joblib.load(os.path.join(parent_path, 'model\%s\%s' % (peptide_length, skb_model_name)))
        clf_model_name = hla + '_clf_model.pkl'
        clf_model = joblib.load(os.path.join(parent_path, 'model\%s\%s' % (peptide_length, clf_model_name)))
        bert_feature = sd_model.transform(bert_feature)
        bert_feature = select_model.transform(bert_feature)
        bert_feature = skb_model.transform(bert_feature)
        y_pre = clf_model.predict(bert_feature)[0]
    except Exception:
        return('Your prediction model is not ready, check the model and path first')

    if y_pre == 1:
        return ('For %s and %s,the result predicted by HLAB is binding' % (allel, peptide))
    else:
        return ('For %s and %s,the result predicted by HLAB is not binding' % (allel, peptide))
