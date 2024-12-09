import pandas as pd

def write_num_samples_csv(args,dataset,fold,sample_number, train):
    if train==True:
        file_name="/media/user/nvme0/ECG-modeling/cpsc_num_samples_train.csv"
    else:
        file_name="/media/user/nvme0/ECG-modeling/cpsc_num_samples_val.csv"
    dicts={
        'labels' : dataset.classes,
        'folds':[fold for i in range(args.num_classes)],
        'sample_number':[sample_number]
    }
    df = pd.DataFrame(dicts)
    if(os.path.isfile(file_name)):
        df.to_csv(file_name, mode='a', index=None, header=None)
    else:
        df.to_csv(file_name,index=None)



dataset = 
label_csv = "/media/user/nvme0/ECG-modeling/cpsc_2018/summary.csv"

df = pd.read_csv(label_csv)
write_num_samples_csv(args, train_dataset,fold_idx,num_samples,train=True)
write_num_samples_csv(args, val_dataset,fold_idx,num_samples,train=False)