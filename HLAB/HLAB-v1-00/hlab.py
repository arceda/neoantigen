# coding=utf-8

# HLAB version 1.0
# Authors: Yaqi Zhang, Fengfeng Zhou
# Contact: FengfengZhou@gmail.com


import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd
from cmd import Cmd
from HLAB.bin import *
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

class Client(Cmd):
    intro = 'Welcome to HLAB! \n' \
            'HLAB is a class I HLA-binding peptide prediction tool by learning the BiLSTM features from the ProtBert-encoded proteins.\n' \
            'Type help or ? to list commands.\n'
    prompt = 'HLAB>'

    def do_query_by_Allel(self, arg):
        'Query the allel supports predicted peptide lengths.The parameter of the function is Allel,You can use this function by <query_by_Allel HLA-A*01:01>.'
        self.query_by_Allel(arg)

    def do_query_all_Allel(self, arg):
        'Query all allel that HLAB supports prediction.You can use this function by <query_all_Allel>'
        self.query_all_Allel(arg)
        
    def do_predict(self, arg):
        'To predict the binding result of HLA-I allele and peptide, the parameter of the function is Allel peptide peptide_length. For example: predict HLA-A*01:01 TSEYHDIMY 9.Please use the query function to query the allele and length predicted by HLAB.'
        self.predict(arg)

    def do_exit(self, _):
        'Exit HLAB.'
        exit(0)
    
    def emptyline(self):
        pass

    def default(self, arg):
        print(u'This command is not available in HLAB')
        
    def query_by_Allel(self, allel):
        df = pd.read_excel(os.path.join(currentPath, 'source/allel.xlsx'))
        support_length = df[df.ALLEL == allel]['peptide_Length'].values
        if support_length.size > 0:
            support_length = support_length.tolist()[0].split(',')
            length_list = list(map(int, support_length))
            print(f'{allel} supports predicted peptide lengths of {length_list}')
        else:
            print(f'HLAB does not support prediction of {allel}')


    def query_all_Allel(self, arg):
        df = pd.read_excel(os.path.join(currentPath, 'source/allel.xlsx'))
        allel_list = df['ALLEL'].tolist()
        print('Allele                         Peptide lengths')
        print('--------------------------------------------------')
        for i in range(len(allel_list)):
            allel = allel_list[i]
            support_length = df[df.ALLEL == allel]['peptide_Length'].values
            support_length = support_length.tolist()[0].split(',')
            length_list = list(map(int, support_length))
            print(f'{allel}                    {length_list}')

    def predict(self, arg):
        arg = arg.split(' ')
        if len(arg) != 3:
            print('Your input does not conform to the specification,\n'
                  ' please type <help predict> to understand the standard input format of the predict function')
        else:
            allel = arg[0]
            peptide = arg[1]
            length = arg[2]
            result = queryAndPredict.predict(allel, peptide, length)       
            print(f'{result}')


if __name__ == '__main__':
    client = Client()
    client.cmdloop()