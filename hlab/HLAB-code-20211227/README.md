# HLAB

HLAB is a class I HLA-binding peptide prediction tool by learning the BiLSTM features from the ProBERT-encoded proteins.

Contact: yqzhang9712@163.com

## requirements

* python==3.6
* cudatoolkit==9.0
* numpy==1.19.2
* pandas==1.1.3
* scikit-learn==0.23.2
* scipy=1.5.2
* tqdm=4.50.2
* pytorch=1.1.0=py3.6_cuda9.0.176_cudnn7.5.1_0
* torchvision=0.3.0=py36_cu9.0.176_1
* h5py==2.10.0
* transformers==4.5.0.dev0    （**pip install -q transformers**）
* umap-learn==0.5.1
* xgboost==1.3.3

## General usage

We provide the **./example** directory and **./example/example_run.py** for users to replicate our prediction process.

The HLAB prediction process is divided into the following steps.

1. Data preprocessing.

The user needs to specify the HLA-I and the length of peptide sequence that they want to predict.From the raw data we provide, extract the raw data of HLA-I and peptides you want. In the examplefile, we extracted the original data of HLA-A*01:01 and the peptide of length 8.

The generated original data file should be with head of **"HLA,peptide,Label,Length"**, For example (example/example_test.csv)

```
HLA,peptide,Label,Length
HLA-A*01:01,LFGRDLSY,1,8
HLA-A*01:01,TDKKTHLY,1,8
HLA-A*01:01,RSDTPLIY,1,8
HLA-A*01:01,NSDLVQKY,1,8
HLA-A*01:01,LSDLLDWK,1,8
HLA-A*01:01,LLQNDGFF,1,8
HLA-A*01:01,DSDMQTLV,1,8
HLA-A*01:01,TDYHVRVY,1,8
HLA-A*01:01,VLDSEGYL,1,8
HLA-A*01:01,SDFHNNRY,1,8
```

---

2. Feature extraction.

In the feature extraction step, we provide a trained feature extraction model in the 'rnnmodels' directory, and users can directly use this model for feature extraction. It should be noted that in this step you need to use gpu to run the feature extraction code.

We also provide the Feature extraction part of the source code for model training and evaluation.

You can run the following bash script to train and evaluate the corresponding model.`bash ./bin/finetune.sh`

3. Feature dimensionality reduction,feature selection and Classification.

We use the **pipeline** to connect the normalization, feature dimensionality reduction, feature selection and classification together.

The file **(./source/parameters.xlsx)** provides the optimal parameter combinations for different HLA-I and peptide binary classification tasks, including the feature dimensionality reduction algorithm UMAP dimensionality parameter after dimensionality reduction (UMAP_params), feature selection algorithm (FS_name), and feature selection percentage parameter ( FS_params).

The user can select the corresponding parameters to pass to the module according to different specific tasks.

4. Get the prediction result.

After running the above three steps, the classification result of your test task will be saved to the result file.
