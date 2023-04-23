import pandas as pd
import os
from bin.feature_extract import Feature_Extract
import bin.feature_selecion as Feature_Selection

currentpath = os.getcwd()
def generate_example(path, hla_type, length):
    df = pd.read_csv(path)
    df_example = df[(df.HLA == hla_type) & (df.Length == length)]
    return df_example

def getFeatureProcessingParameters(hla_type, length):
    df = pd.read_excel('%s/source/parameters.xlsx' % currentpath)
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


if __name__ == '__main__':
    # The HLA-I and peptide lengths supported by HLAB provided by our supplementary file can be arbitrarily specified
    hla_type = 'HLA-A*01:01'
    peptide_length = 8

    # Generate example dataset.
    df_example = generate_example('%s/dataset/train_data.csv' % currentpath, hla_type, peptide_length)
    df_example.to_csv('%s/example/example_train.csv' % currentpath, index=False)
    df_example = generate_example('%s/dataset/valid_data.csv' % currentpath, hla_type, peptide_length)
    df_example.to_csv('%s/example/example_valid.csv' % currentpath, index=False)
    df_example = generate_example('%s/dataset/test_data.csv' % currentpath, hla_type, peptide_length)
    df_example.to_csv('%s/example/example_test.csv' % currentpath, index=False)

    # Extract features using the trained ProBERT+BiLSTM model.
    # resourece warning!!! use GPU!!
    FE = Feature_Extract("train", "./rnnmodels", 51)
    FE.embedding()
    FE = Feature_Extract("valid", "./rnnmodels", 51)
    FE.embedding()
    FE = Feature_Extract("test", "./rnnmodels", 51)
    FE.embedding()

    #Feature dimensionality reduction, feature selection and Classification
    params_list = getFeatureProcessingParameters(hla_type, peptide_length)

    train = pd.read_hdf('%s/example/train_embed.h5' % currentpath)
    valid = pd.read_hdf('%s/example/valid_embed.h5' % currentpath)
    test = pd.read_hdf('%s/example/test_embed.h5' % currentpath)  # 读取h5
    train_= [0, 0]
    val_= [0, 0]
    test_ = [0, 0]
    test_[0], test_[1] = test.values[:, 0:-1], test.values[:, -1]
    val_[0], val_[1] = valid.values[:, 0:-1], valid.values[:, -1]
    train_[0], train_[1] = train.values[:, 0:-1], train.values[:, -1]
    result = Feature_Selection.fs(train_, val_, test_, params_list)
    result = list(map(int, result))
    result_df = pd.read_csv('%s/example/example_test.csv' % currentpath)
    result_df['Predict'] = result
    # save prediction result to predict_result.xlsx
    result_df.to_excel('%s/example/predict_result.xlsx' % currentpath)









