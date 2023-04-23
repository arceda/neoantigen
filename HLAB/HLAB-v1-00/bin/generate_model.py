# encoding:utf-8
import pandas as pd
import umap
from scipy import stats
from sklearn.feature_selection import SelectKBest
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
import joblib
from HLAB.bin import feature_generate as fe
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


def model_save(train, val, test, params_list, hla, peptide_length):
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
    joblib.dump(sd, 'model/%s/%s_sd_model.pkl' % (peptide_length, hla))

    #  Feature dimensionality reduction
    select = umap.UMAP(n_components=umap_params)
    select.fit(train[0], train[1])
    X_train = select.transform(train[0])
    X_val = select.transform(val[0])
    X_test = select.transform(test[0])
    joblib.dump(select, 'model/%s/%s_select_model.pkl' % (peptide_length, hla))

    #  Feature selection
    skb = SelectKBest(score_func=fs_name, k=int(X_train.shape[1]))
    skb.fit(X_train, train[1])
    skb.k = int(X_train.shape[1] * fs_params)
    X_train_i = skb.transform(X_train)
    X_val_i = skb.transform(X_val)
    X_test_i = skb.transform(X_test)
    joblib.dump(skb, 'model/%s/%s_skb_model.pkl' % (peptide_length, hla))

    clf = clfs[clf_name]
    clf.fit(X_train_i, train[1])
    scorce = clf.score(X_val_i, val[1])
    joblib.dump(clf, 'model/%s/%s_clf_model.pkl' % (peptide_length, hla))


def getFeatureProcessingParameters(hla_type, length):
    dir_path = os.path.join(parent_path, 'source/parameters.xlsx')
    df = pd.read_excel(dir_path)
    df = df[(df.HLA == hla_type) & (df.Length == length)]
    classifier = df.Classifier.values[0]
    fs_name = df.FS_name.values[0]
    UMAP_params = df.UMAP_params.values[0]
    fs_params = df.FS_params.values[0]
    return [
        classifier,
        fs_name,
        UMAP_params,
        fs_params,
    ]

def getFeature(hla, peptide_length, type):
    data_path = os.path.join(parent_path, 'data')
    data_path = os.path.join(data_path, type + '_data.csv')
    model_path = os.path.join(os.path.dirname(parent_path), 'HLAB-code-20211227')
    model_path = os.path.join(model_path, 'rnnmodels')
    df = pd.read_csv(data_path)
    data_df = df[(df.HLA == hla) & (df.Length == peptide_length)]
    print(data_df.info)
    feature_df = fe.Feature_Extract(data_df, type, model_path,).embedding()
    return feature_df


if __name__ == '__main__':


    dict = {
        "8": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_03_01", "HLA-A_11_01", "HLA-A_24_02", "HLA-A_29_02", "HLA-B_07_02",
              "HLA-B_08_01", "HLA-B_13_02", "HLA-B_14_02", "HLA-B_15_01", "HLA-B_18_01", "HLA-B_18_03", "HLA-B_27_05",
              "HLA-B_27_09", "HLA-B_35_01", "HLA-B_37_01", "HLA-B_39_01", "HLA-B_39_24", "HLA-B_40_01", "HLA-B_40_02",
              "HLA-B_44_02", "HLA-B_44_03", "HLA-B_46_01", "HLA-B_49_01", "HLA-B_51_01", "HLA-B_51_08", "HLA-B_52_01",
              "HLA-B_54_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01", "HLA-C_01_02", "HLA-C_02_02", "HLA-C_03_03",
              "HLA-C_03_04", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02", "HLA-C_07_01", "HLA-C_07_02", "HLA-C_07_04",
              "HLA-C_08_02", "HLA-C_12_03", "HLA-C_14_02", "HLA-C_15_02", "HLA-C_16_01", "HLA-C_17_01"],
        "9": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_02_02", "HLA-A_02_03", "HLA-A_02_04", "HLA-B_07_02", "HLA-B_08_01",
              "HLA-B_13_02", "HLA-B_14_01", "HLA-B_14_02", "HLA-B_15_01", "HLA-B_15_02", "HLA-B_15_03", "HLA-B_15_09",
              "HLA-B_15_11", "HLA-B_15_17", "HLA-B_15_18", "HLA-B_15_42", "HLA-B_18_01", "HLA-B_18_03", "HLA-B_27_01",
              "HLA-B_27_02", "HLA-B_27_03", "HLA-B_27_04", "HLA-B_27_05", "HLA-B_27_06", "HLA-B_27_07", "HLA-B_27_08",
              "HLA-B_27_09", "HLA-B_27_20", "HLA-B_35_01", "HLA-B_35_03", "HLA-B_35_08", "HLA-B_37_01", "HLA-B_38_01",
              "HLA-B_39_01", "HLA-B_39_06", "HLA-B_39_24", "HLA-C_01_02", "HLA-C_02_02", "HLA-C_03_03", "HLA-C_03_04",
              "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02", "HLA-C_07_01", "HLA-C_07_02", "HLA-C_07_04", "HLA-C_08_02",
              "HLA-C_12_03", "HLA-C_14_02", "HLA-C_15_02", "HLA-C_16_01", "HLA-C_17_01"],
        "10": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_02_02", "HLA-A_02_03", "HLA-A_02_04", "HLA-A_02_05", "HLA-A_02_06",
               "HLA-A_02_07", "HLA-A_02_17", "HLA-A_03_01", "HLA-A_11_01", "HLA-A_23_01", "HLA-A_24_02", "HLA-A_24_06",
               "HLA-A_26_01", "HLA-A_29_02", "HLA-A_30_01", "HLA-A_30_02", "HLA-A_31_01", "HLA-A_32_01", "HLA-A_33_01",
               "HLA-A_68_01", "HLA-A_68_02", "HLA-A_69_01", "HLA-B_07_02", "HLA-B_08_01", "HLA-B_13_02", "HLA-B_14_02",
               "HLA-B_15_01", "HLA-B_18_01", "HLA-B_27_01", "HLA-B_27_02", "HLA-B_27_03", "HLA-B_27_04", "HLA-B_27_05",
               "HLA-B_27_06", "HLA-B_27_07", "HLA-B_27_08", "HLA-B_27_09", "HLA-B_35_01", "HLA-B_35_03", "HLA-B_35_08",
               "HLA-B_37_01", "HLA-B_39_01", "HLA-B_40_01", "HLA-B_40_02", "HLA-B_41_01", "HLA-B_44_02", "HLA-B_44_03",
               "HLA-B_44_27", "HLA-B_45_01", "HLA-B_46_01", "HLA-B_49_01", "HLA-B_50_01", "HLA-B_51_01", "HLA-B_53_01",
               "HLA-B_54_01", "HLA-B_56_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01", "HLA-C_01_02", "HLA-C_02_02",
               "HLA-C_03_03", "HLA-C_03_04", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02", "HLA-C_07_01", "HLA-C_07_02",
               "HLA-C_07_04", "HLA-C_08_02", "HLA-C_14_02", "HLA-C_16_01"],
        "11": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_02_03", "HLA-A_02_04", "HLA-A_02_05", "HLA-A_02_07", "HLA-A_03_01",
               "HLA-A_11_01", "HLA-A_23_01", "HLA-A_24_02", "HLA-A_24_06", "HLA-A_29_02", "HLA-A_31_01", "HLA-A_32_01",
               "HLA-A_68_01", "HLA-A_68_02", "HLA-B_07_02", "HLA-B_08_01", "HLA-B_15_01", "HLA-B_27_01", "HLA-B_27_02",
               "HLA-B_27_03", "HLA-B_27_04", "HLA-B_27_05", "HLA-B_27_06", "HLA-B_27_07", "HLA-B_27_08", "HLA-B_27_09",
               "HLA-B_35_01", "HLA-B_35_03", "HLA-B_35_08", "HLA-B_37_01", "HLA-B_39_01", "HLA-B_40_01", "HLA-B_40_02",
               "HLA-B_44_02", "HLA-B_44_03", "HLA-B_45_01", "HLA-B_46_01", "HLA-B_49_01", "HLA-B_51_01", "HLA-B_54_01",
               "HLA-B_56_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01", "HLA-C_01_02", "HLA-C_02_02", "HLA-C_03_03",
               "HLA-C_03_04", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02", "HLA-C_07_01", "HLA-C_07_02", "HLA-C_08_02",
               "HLA-C_16_01"],
        "12": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_03_01", "HLA-A_11_01", "HLA-A_24_02", "HLA-A_29_02", "HLA-A_31_01",
               "HLA-A_68_01", "HLA-A_68_02", "HLA-B_07_02", "HLA-B_08_01", "HLA-B_15_01", "HLA-B_27_01", "HLA-B_27_02",
               "HLA-B_27_03", "HLA-B_27_05", "HLA-B_27_07", "HLA-B_27_08", "HLA-B_27_09", "HLA-B_35_01", "HLA-B_40_01",
               "HLA-B_40_02", "HLA-B_44_02", "HLA-B_44_03", "HLA-B_51_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01",
               "HLA-C_01_02", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02", "HLA-C_07_01"],
        "13": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_03_01", "HLA-A_11_01", "HLA-A_24_02", "HLA-A_29_02", "HLA-A_31_01",
               "HLA-A_68_02", "HLA-B_07_02", "HLA-B_15_01", "HLA-B_27_01", "HLA-B_27_02", "HLA-B_27_05", "HLA-B_27_08",
               "HLA-B_27_09", "HLA-B_35_01", "HLA-B_44_02", "HLA-B_51_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01",
               "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02"],
        "14": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_24_02", "HLA-A_68_02", "HLA-B_07_02", "HLA-B_15_01", "HLA-B_27_05",
               "HLA-B_27_09", "HLA-B_35_01", "HLA-B_57_01", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02"]}

    for key in ["8", "9", "10", "11", "12", "13", "14"]:
        for hla in dict[key]:
            peptide_length = int(key)
            hla_type = hla[0:5] + '*' + hla[6:8] + ':' + hla[9:11]
            train = getFeature(hla_type, peptide_length, 'train')
            valid = getFeature(hla_type, peptide_length, 'valid')
            test = getFeature(hla_type, peptide_length, 'test')
            train_ = [0, 0]
            val_ = [0, 0]
            test_ = [0, 0]
            test_[0], test_[1] = test.values[:, 0:-1], test.values[:, -1]
            val_[0], val_[1] = valid.values[:, 0:-1], valid.values[:, -1]
            train_[0], train_[1] = train.values[:, 0:-1], train.values[:, -1]
            params_list = getFeatureProcessingParameters(hla_type, peptide_length)
            model_save(train_, val_, test_, params_list, hla, peptide_length)