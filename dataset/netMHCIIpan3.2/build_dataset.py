# Author: Vicente

# Este script genera un csv para el entrenamiento de BERTMHC, 
# va a consiederar las muestras de netMHCIIpan3.2.
# Todas las pseudosecuencias estan netMHCIIpan4.1.
# ala parecer esta basa de daos es mas grande que la utilizada en BERTMHC.

import pandas as pd  
import numpy as np

# reading netMHCIIpan3.2 dataset ########################################################
# tenemos 5 splits
train = pd.DataFrame() 
test = pd.DataFrame() 
for i in range(1,6):
    split_train = pd.read_csv(f"train{i}.txt", header=None, delim_whitespace=True)
    split_test = pd.read_csv(f"test{i}.txt", header=None, delim_whitespace=True)

    train = pd.concat([train, split_train], ignore_index=True, axis=0)
    test = pd.concat([test, split_test], ignore_index=True, axis=0)

   


# reading pseudosequences ###############################################################
pseudo_sequences = pd.read_csv( f"../netMHCIIpan4.1/pseudosequence.2016.all.X.dat", 
                                index_col=0, header=None, delim_whitespace=True)
#print(pseudo_sequences)
#print(pseudo_sequences.loc["DRB1_0101"])
#print(pseudo_sequences.loc["BoLA-DQA2101-DQB0901"])


# create dataset for BERTMHC ############################################################
train.rename(columns={0:'peptide', 1:'label', 2:'mhc_type'},  inplace=True)
train['masslabel'] = train.apply(lambda row: (1 if row['label'] > 0.426 else 0), axis=1)
train['mhc'] = train.apply(lambda row: ( pseudo_sequences.loc[row['mhc_type']] ), axis=1)
train_shuffle = train.sample(frac=1, random_state=42).reset_index(drop=True)

val_count = int(train_shuffle.shape[0]*0.15)
print(val_count)
eval = train_shuffle.iloc[:val_count, :]
train = train_shuffle.iloc[val_count:, :] 

train.to_csv("train.csv", index=False)
eval.to_csv("eval.csv", index=False)

test.rename(columns={0:'peptide', 1:'label', 2:'mhc_type'},  inplace=True)
test['masslabel'] = test.apply(lambda row: (1 if row['label'] > 0.426 else 0), axis=1)
test['mhc'] = test.apply(lambda row: ( pseudo_sequences.loc[row['mhc_type']] ), axis=1)
test.to_csv("test.csv", index=False)


