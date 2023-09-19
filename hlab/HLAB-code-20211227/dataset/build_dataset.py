# Author: Vicente

# Este script genera un csv para el entrenamiento de BERTMHC, 
# va a consiederar las muestras de netMHCIIpan3.2.
# Todas las pseudosecuencias estan netMHCIIpan4.1.
# ala parecer esta basa de daos es mas grande que la utilizada en BERTMHC.

import pandas as pd  
import numpy as np
import sys
   
train = pd.read_csv(f"train_data.csv")
test = pd.read_csv(f"test_data.csv")
val = pd.read_csv(f"valid_data.csv")

print(train.shape)
print(test.shape)
print(val.shape)



# reading pseudosequences ###############################################################
pseudo_sequences = pd.read_csv( f"MHC_pseudo.dat", index_col=0, delim_whitespace=True)
#print(pseudo_sequences)
print(pseudo_sequences.loc["HLA-A01:06"])



# create dataset  ############################################################
train['mhc'] = train.apply(lambda row: ( pseudo_sequences.loc[ row['HLA'].replace("*", "") ] ), axis=1)
test['mhc'] = test.apply(lambda row: ( pseudo_sequences.loc[row['HLA'].replace("*", "") ] ), axis=1)
val['mhc'] = val.apply(lambda row: ( pseudo_sequences.loc[row['HLA'].replace("*", "") ] ), axis=1)


train.to_csv("hlab_train.csv", index=False)
test.to_csv("hlab_test.csv", index=False)
val.to_csv("hlab_val.csv", index=False)


