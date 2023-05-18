###Data description file###

test_data.csv: independent test data
valid_data.csv: validation data,25% of the train_all data
train_data.csv: 75% of the train_all data

train_all.csv: valid_data + train_data, total training data

train_pos.csv: The original training set with only positive sample
test_pos.csv: The original test set with only positive sample
train_neg.csv:the generated training set with only negative sample
test_neg.csv: the generated test set with only negative sample
