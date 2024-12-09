import pandas as pd
import numpy as np
import os


dataset= 'cpsc_2018'

if(dataset=='cpsc_2018'):
    num_samples_file_train = "/media/user/nvme0/ECG-modeling/cpsc_num_samples_train.csv"
    num_samples_file_test = "/media/user/nvme0/ECG-modeling/cpsc_num_samples_val.csv"
    classes=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
elif(dataset=='ptb-xl'):
    num_samples_file_train = "/media/user/nvme0/ECG-modeling/ptb_num_samples_train.csv"
    num_samples_file_test = "/media/user/nvme0/ECG-modeling/ptb_num_samples_val.csv"
    classes=['NORM','MI','STTC','CD','HYP']
elif (dataset == 'shaoxing-ninbo'):
    num_samples_file_train = "/media/user/nvme0/ECG-modeling/shaoxing_num_samples_train.csv"
    num_samples_file_test = "/media/user/nvme0/ECG-modeling/shaoxing_num_samples_val.csv"
    classes=['1AVB', '2AVB', '2AVB1', '2AVB2', '3AVB', 'ABI', 'ALS', 'APB', 'AQW', 'ARS', 'AVB', 
                        'CCR', 'CR', 'ERV', 'FQRS', 'IDC', 'IVB', 'JEB', 'JPT', 'LBBB', 'LBBBB', 
                        'LFBBB', 'LVH', 'LVQRSAL', 'LVQRSCL', 'LVQRSLL', 'MI', 'MIBW', 'MIFW', 'MILW', 
                        'MISW', 'PRIE', 'PWC', 'QTIE', 'RAH', 'RBBB', 'RVH', 'STDD', 'STE', 'STTC', 'STTU', 
                        'TWC', 'TWO', 'UW', 'VB', 'VEB', 'VFW', 'VPB', 'VPE', 'VET', 'WAVN', 'WPW', 'SB', 'SR', 
                        'AFIB', 'ST', 'AF', 'SA', 'SVT', 'AT', 'AVNRT', 'AVRT', 'SAAWR']


model_name="Residual_ConvTransformer_2024-08-15-19:44_all_42"
result_dir = f'/media/user/nvme0/ECG-modeling/{dataset}/results/'
summary=f'/media/user/nvme0/ECG-modeling/{dataset}/balanced_accuracy.csv'

all_label_accuracy = f'/media/user/nvme0/ECG-modeling/{dataset}/all_label_accuracy.csv'


for i in range(2):
    bal_acc_list=[]
    acc_list=[]
    for fold_idx in range(9):
        if(i==0):
            file_name=result_dir+model_name+f'_train_{fold_idx}_folds.csv'
            num_samples_file = num_samples_file_train
        else:
            file_name=result_dir+model_name+f'_val_{fold_idx}_folds.csv'
            num_samples_file = num_samples_file_test
        
        df1 = pd.read_csv(file_name)
        acc = df1['accuracy'][1:].values.tolist()
        acc_list.append(np.array(acc))
        df2 = pd.read_csv(num_samples_file)
        num_sample = df2[df2['folds']==fold_idx]['sample_number'].values.tolist()   
        
        score = np.array(acc)*np.array(num_sample)/sum(num_sample)
        balance_accuracy = sum(score)
        bal_acc_list.append(balance_accuracy)
    avg_bal_acc=sum(bal_acc_list)/len(bal_acc_list)
    dicts={
        'model_name':[model_name],
        'train_test':['train' if i==0 else 'test'],
        'balanced_accuracy':[avg_bal_acc]
    }

    print(avg_bal_acc)
    df=pd.DataFrame(dicts)
    if(os.path.isfile(summary)):
        df.to_csv(summary, mode='a', index=None, header=None)
    else:
        df.to_csv(summary,index=None)
    avg_per_label_acc=sum(acc_list)/len(acc_list)
    column_list=['model_name','train_test']
    train_test='train' if i==0 else 'test'
    data_list=[model_name, train_test]
    dict_label={
        'model_name':[model_name],
        'train_test':[train_test]
    }
    df_label=pd.DataFrame(dict_label)
    
    for j,name in enumerate(classes):
        df_label.insert(2,name,avg_per_label_acc[j])
    if(os.path.isfile(all_label_accuracy)):
        df_label.to_csv(all_label_accuracy, mode='a', index=None, header=None)
    else:
        df_label.to_csv(all_label_accuracy,index=None)    
