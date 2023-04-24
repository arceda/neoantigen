# Author: Vicente

# Este script genera una bd pequeña de 134281 muestras, al pareer BERTMHC uso esto xq 
# la bd origianl tiene mas de 500k muestras

import pandas as pd  

db = pd.read_csv( "test.csv")
db = db.sample(frac=1, random_state=42).reset_index(drop=True)
rows = db.shape[0]
train = db.iloc[:int(rows*0.8), :]
eval = db.iloc[int(rows*0.8):int(rows*0.9), :]
test = db.iloc[int(rows*0.9):, :]

print(train.shape)
print(eval.shape)
print(test.shape)

train.to_csv("train_mini.csv", index=False)
eval.to_csv("eval_mini.csv", index=False)
test.to_csv("test_mini.csv", index=False)


