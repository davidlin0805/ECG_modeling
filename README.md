# Development and Deployment of Generalizable Foundation Models for 12-lead ECG Classification

## Requirement
### Dataset
#### cpsc_2018
CPSC2018 is the dataset of 2018 PhysioNet/CinC Challenge. The challenge aims to encourage the
development of algorithms to identify the rhythm/morphology abnormalities from 12-lead Electrocardiograms (ECGs).
#### Shaoxing-Ninbo 
This dataset for 12-lead electrocardiogram signals was created under the auspices of
Chapman University, Shaoxing People’s Hospital (Shaoxing Hospital Zhejiang University School of Medicine), and
Ningbo First Hospital. 
#### PTB-XL
The PTB-XL ECG dataset is a large dataset of 21799 clinical 12-lead ECGs from 18869 patients of 10
second length. Where 52% are male and 48% are female with ages covering the whole range from 0 to 95 years. 

## Run
The files collectively define the infrastructure for training, validating, and performing inference on ECG classification models using various deep learning architectures. \
The models.py file defines different neural network models (LSTM, GRU, ResNet, and Transformer variants). These models are tailored for tasks like ECG classification, handling temporal dependencies and feature extraction in sequence data.\
The K_fold_cross.py script manages the training and evaluation processes, leveraging cross-validation to assess model performance across multiple folds, and tracks metrics. \
The dataset.py file defines PyTorch Dataset classes to handle different ECG datasets (e.g., CPSC-2018, PTB-XL, Shaoxing-Ninbo), supporting data augmentation and transformation.\
The inference.py is designed for performing inference on new ECG data, using trained models to predict outcomes and track inference performance.\
The train_tetst.py is for training and testing the dataset according to the percentage you want.\
The  foundation_model.py is for training the foundation models. It trains three models at the same time then aggregate to produce global model. \

### Preprocessing
To get labels.csv for training
```sh
$ python utils/data_preprocessor/datapreprocess.py 
```
### K_fold Cross validation
The configuration can be changed in config.txt
```sh
$ python K_fold_cross.py -config_file configuration/config.txt
```
Or change the arguments in the code.
```sh
$ python K_fold_cross.py  
```
#### Train Foundation Model
```sh
$ python foundation_model.py
```
#### Train_test.py
Test for model compression result or train a model from scratch. Enable the arguments of quantize or prune in train_test.py for compression results.
```sh
$ python train_test.py
```
