# este script agrega la z para que cada secuenia tenga el mismo tamaño y aplica onehot encoding.
# solo se utiliza para entender el procedimiento aplicado por le paper
import numpy as np
import pandas as pd
import random as rnd
from sklearn.model_selection import train_test_split
import os

# insert "Z" in order to get same lenght sequences
def transformEL(dataset):
    dataset = dataset.reset_index(drop=True)
    peptide=dataset.Peptide
    peptide2list=peptide.tolist()
    for i in range(len(peptide)):
        if len(peptide2list[i]) < 11:
            n1 = len(peptide2list[i]) // 2
            n2 = 11 - len(peptide2list[i])
            peptide2list[i] = peptide2list[i][:n1] + 'Z'*n2 + peptide2list[i][n1:]     #将小于11个氨基酸的peptide在中间插空补齐
        else:
            peptide2list[i] = peptide2list[i][:11]

    del dataset['Peptide']
    peptides = pd.DataFrame(peptide2list,columns=['Peptide'])
    dataset.insert(0,'Peptide',peptides.pop('Peptide'))
    return dataset

def Pept_OneHotMap(peptideSeq):
    """ maps amino acid into its numerical index
    USAGE
    Pept_OneHotMap('A')
    array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    # integer encode input data
    integer_encoded=[char2int[char] for char in peptideSeq]
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
    	letter = [0 for _ in range(len(allSequences))]
    	letter[value] = 1
    	onehot_encoded.append(letter)
    return np.asarray(onehot_encoded)


##################################################################
###all the possible sequence letters
allSequences = 'ACEDGFIHKMLNQPSRTWVYZ'
# Establish a mapping from letters to integers
char2int = dict((c, i) for i, c in enumerate(allSequences))
##################################################################



traindata = os.path.join("./../DATA", "train_data", "A0101__" )
print("traindata name: ", traindata)
df_train = pd.read_csv(traindata, header=0)

#print(df_train, df_train.shape)
df_train = df_train[df_train.Peptide.str.contains('X') == False]
df_train = df_train[df_train.Peptide.str.contains('B') == False]
df_train = df_train[df_train.Peptide.str.contains('U') == False]
print("train data original:\n", df_train, "\n")



# onehot matrix
train_data=transformEL(df_train)

print("train_data after transformEL:\n", train_data, "\n")




trainMatrix = np.empty((0, 11,len(allSequences)), int)      
for num in range(len(train_data.Peptide)):
    if num%1000 == 0:
        print(train_data.Peptide.iloc[num],num)
    trainMatrix = np.append(trainMatrix, [Pept_OneHotMap(train_data.Peptide.iloc[num])], axis=0)
allele_name = train_data['HLA'][0]
assert (trainMatrix.shape[0] == train_data.shape[0])

print("train_data after onehot:\n", trainMatrix, "\n")

print(trainMatrix.shape)
