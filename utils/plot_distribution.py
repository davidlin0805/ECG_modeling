import matplotlib.pyplot as plt
import os
import pandas as pd
import wfdb
import numpy as np
import seaborn as sns
import tqdm

def write_feature_val(classes,df,data_dir,result_dir):
    min_val=0
    max_val=0
    min_list=[]
    max_list=[]
    avg_list=[]
    std_list=[]
    data_list=[]
    labels_samples=[]
    class_list=[]
    for label in classes:
        print(label)
        if(len(df[df[label]==1])>0):
            patient_ids = df[df[label]==1]['patient_id']
            for patient_id in patient_ids:
                if(dataset_name=='ptb-xl'):
                    data,_=wfdb.rdsamp(os.path.join(data_dir, patient_id[-8:]))
                else:
                    data,_=wfdb.rdsamp(os.path.join(data_dir, patient_id))
                data_flattened = data.flatten()  # Flatten the 2D array into a 1D array
                data_list.append(data_flattened)
            
            labels_samples.append(len(df[df[label]==1]))
            class_list.append(label)
            ecg_data=np.array(data_flattened)
            std=np.std(ecg_data, ddof=1)
            min_val=np.min(ecg_data)
            max_val=np.max(ecg_data)
            mean_val=np.mean(ecg_data)
            min_list.append(min_val)
            max_list.append(max_val)
            avg_list.append(mean_val)
            std_list.append(std)
    dicts={
        "class":class_list,
        "number":labels_samples,
        "min":min_list,
        "max":max_list,
        "avg":avg_list,
        "std":std_list
    }
    df2=pd.DataFrame(dicts, index=None)
    df2.to_csv(f'{result_dir}{dataset_name}_class_distribution.csv')
    exit()
        
dataset_name = "cpsc_2018"
data_dir = f'/media/user/nvme0/ECG-modeling/{dataset_name}/datas/'
result_dir = f'/media/user/nvme0/ECG-modeling/{dataset_name}/plots/'
label_csv = os.path.join(data_dir, 'labels.csv')

if(dataset_name=='cpsc_2018'):
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
elif(dataset_name=='ptb-xl'):
    classes = ['NORM','MI','STTC','CD','HYP']
elif(dataset_name=='shaoxing-ninbo'):
    classes = ['1AVB', '2AVB', '2AVB1', '2AVB2', '3AVB', 'ABI', 'ALS', 'APB', 'AQW', 'ARS', 'AVB', 
                        'CCR', 'CR', 'ERV', 'FQRS', 'IDC', 'IVB', 'JEB', 'JPT', 'LBBB', 'LBBBB', 
                        'LFBBB', 'LVH', 'LVQRSAL', 'LVQRSCL', 'LVQRSLL', 'MI', 'MIBW', 'MIFW', 'MILW', 
                        'MISW', 'PRIE', 'PWC', 'QTIE', 'RAH', 'RBBB', 'RVH', 'STDD', 'STE', 'STTC', 'STTU', 
                        'TWC', 'TWO', 'UW', 'VB', 'VEB', 'VFW', 'VPB', 'VPE', 'VET', 'WAVN', 'WPW', 'SB', 'SR', 
                        'AFIB', 'ST', 'AF', 'SA', 'SVT', 'AT', 'AVNRT', 'AVRT', 'SAAWR']

lead_name=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

df = pd.read_csv(label_csv)
labels_samples=[]
labels_samples_all=[]
label_list=[]
for label in classes:
    if(len(df[df[label]==1])>0):
        label_list.append(label)
        labels_samples.append(len(df[df[label]==1]))
    labels_samples_all.append(len(df[df[label]==1]))
plt.bar(label_list,labels_samples) 
plt.xlabel("class") 
plt.ylabel("number of samples")
plt.xticks(rotation=90)
plt.savefig(f'{result_dir}{dataset_name}_class_distribution.png') 

write_feature_val(classes, df, data_dir,result_dir)



ecg_data_list=[]
mean_of_one_sample=[]

len_of_dataset=df.shape[0]

n_batches = 200
batch_size = len_of_dataset // n_batches
n_bin=50

# Initialize list to hold histogram data for each channel
histograms = [np.zeros(n_bin) for _ in range(12)]

# Process data in batches
for i in range(0, len_of_dataset, batch_size):
    batch_row =df['patient_id'][i:i+batch_size] 
    name=[]
    batch_data=[]
    print(i)
    for row in batch_row:
        if(dataset_name=='ptb-xl'):
            data,_=wfdb.rdsamp(os.path.join(data_dir, row[-8:]))
        else:
            data,_=wfdb.rdsamp(os.path.join(data_dir, row))
        batch_data.append(data)    

    for j in range(12):
        channel_data = np.concatenate([sample[:, j] for sample in batch_data])
        hist, bin_edges = np.histogram(channel_data, bins=n_bin, range=(-0.5, 0.5))
        histograms[j] += hist
    


for i in range(12):
    plt.figure(figsize=(20, 15))
    plt.bar(bin_edges[:-1], histograms[i], width=np.diff(bin_edges),edgecolor='black')
    
    plt.xlabel('Value', fontsize=30)
    plt.ylabel('Frequency',fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()

    plt.savefig(f'{result_dir}{dataset_name}_values_distribution_lead{i}.png') 


