import json
import os
from collections import OrderedDict
from bin.negpeptide_all import generateneg
import pandas as pd
from sklearn.model_selection import train_test_split

currentpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def check():
    with open('%s/source/lenghHLA.json' % currentpath) as f:
        dict_lenHLA = json.load(f)
    HLA_seq = pd.read_csv('%s/source/MHC_pseudo.dat' % currentpath, sep='\t')
    HLA_List = HLA_seq.HLA.tolist()
    print(HLA_List)
    for len in ["8", "9", "10", "11", "12", "13", "14"]:
        lenghHLAavailable = dict_lenHLA[len]
        for eachHLA in lenghHLAavailable:
            eachHLA = eachHLA[0:5] + eachHLA[6:]
            if eachHLA in HLA_List:
                continue
            else:
                print(len + eachHLA + "non")


def process_data(file):
    HLA, pep, Len, Label = [], [], [], []
    with open(file, 'rt') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Length:"):
                length = line[8:].strip()
                continue
            if line.startswith("HLA-"):
                HLA_ = line.strip()
                continue
            Len.append(length)
            HLA.append(HLA_)
            pep.append(line.strip())
            Label.append("1")
    f.close()
    data_frame = pd.DataFrame({'HLA': HLA, 'peptide': pep, 'Length': Len, 'Label': Label})  # 创建一个4列的空dataframe
    return data_frame


def generate_neg(file):
    '''
    generate neg_pep
    :param file:
    :return:
    '''
    data_frame = pd.read_csv(file)  # pos
    with open('%s/source/lenghHLA.json' % currentpath) as f:
        dict_lenHLA = json.load(f)
    HLA, pep, Len, Label = [], [], [], []
    for length in ["8", "9", "10", "11", "12", "13", "14"]:
        HLA_List = dict_lenHLA[length]
        len_count = 0
        len_count_1 = 0
        for each_HLA in HLA_List:
            print(each_HLA, length)
            count_pos = data_frame[
                (data_frame['HLA'] == each_HLA) & (data_frame['Length'] == int(length))].Index.count()  # pos_pep count
            pep_list = data_frame[(data_frame['HLA'] == each_HLA) & (data_frame['Length'] == int(length))][
                'peptide'].tolist()
            label_list = data_frame[(data_frame['HLA'] == each_HLA) & (data_frame['Length'] == int(length))][
                'Label'].tolist()
            dict_pospep = OrderedDict()
            for items in enumerate(pep_list):
                dict_pospep[items[1]] = label_list[items[0]]
            dict_negpep = generateneg(int(length), dict_pospep)['neg']  #neg_pep
            count_neg = len(dict_negpep)  # neg_pep count
            for negpep in dict_negpep:
                Len.append(int(length))
                HLA.append(each_HLA)
                pep.append(negpep)
                Label.append("-1")
            print(count_neg, count_pos)
            len_count = len_count + count_pos
            len_count_1 = len_count_1 + count_neg
    df_neg = pd.DataFrame({'HLA': HLA, 'peptide': pep, 'Length': Len, 'Label': Label})  # 创建一个4列的空dataframe
    return df_neg
    # df_neg.to_csv("train_neg.csv")
    # df_neg.to_csv("test_neg.csv")


def concat(file1, file2):
    '''
    concat neg and pos
    :param file1:
    :param file2:
    :return:
    '''
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat((df1, df2), ignore_index=True)
    df.drop(axis=1, columns=['Index'], inplace=True)
    df.to_csv("test.scv")


def label_trans(file):
    '''
    label 1->1
         -1->0
    :param file:
    :return:
    '''
    df = pd.read_csv(file)
    df['Label'] = df['Label'].apply(lambda x: 1 if int(x) == 1 else 0)
    df.to_csv("train.csv", index=False)


def split_train_valid_data(file):
    '''
    split original train file to new train and new valid with 3:1
    :param file:
    :return:
    '''
    df = pd.read_csv(file)
    train, valid = train_test_split(df, test_size=0.25, random_state=1, stratify=df['Label'])
    train = pd.DataFrame(train, columns=['HLA', 'peptide', 'Label', 'Length'])
    valid = pd.DataFrame(valid, columns=['HLA', 'peptide', 'Label', 'Length'])
    train.to_csv('tain.csv', index=False)
    valid.to_csv('dev.csv', index=False)

