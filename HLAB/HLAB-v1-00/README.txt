# HLAB version 1.0
# Authors: Yaqi Zhang, Fengfeng Zhou
# Contact: FengfengZhou@gmail.com

HLAB is a class I HLA-binding peptide prediction tool by learning the BiLSTM features from the ProtBert-encoded proteins.

Email: FengfengZhou@gmail.com or ffzhou@jlu.edu.cn 

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
First, since the prediction model in the experiment needs to occupy 5 GB of memory, 
in order to save the user's download time, we provide a local download method. You 
need to make sure you have enough space and run **python generate_model.py** to get the
classification model for prediction.

After downloading the model you can run **python hlab.py** using HLAB's predict function.
Here are some of the features provided by HLAB.
* Query all allel that HLAB supports prediction.You can use this function by **query_all_Allel**
* Query the allel supports predicted peptide lengths.The parameter of the function is Allel,You can use this function by <query_by_Allel HLA-A*01:01>
* To predict the binding result of HLA-I allele and peptide, the parameter of the function is Allel peptide peptide_length. For example: predict HLA-A*01:01 TSEYHDIMY 9.
Please use the query function to query the allele and length predicted by HLAB.


