import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import ECGDataset,PTBDataset,shaoxingDataset
import time
from utils.utils import split_data
from K_fold_cross import model_initialize
import pandas as pd
from tqdm import tqdm
from utils.models.models import cpsc_champion

def test_model_inference_speed(args,loader, model, device):
    model.eval()
    time_list=[]
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(loader)):
            if(args.dataset_name=='shaoxing-ninbo'):
                data=torch.nan_to_num(data)
            data = data.to(device)
            if(args.quantize):
                data=data.half()            
            start_time = time.time()
            if(args.quantize):
                if(args.model_used=='Residual_Conv_GRU' or args.model_used=='ResU_Dense'):
                    model(data,quantize=True) #need to conert h0 
                else:
                    model(data)
            else:                
                model(data)
            end_time = time.time()
            time_list.append(end_time-start_time)
    avg_inference_time = sum(time_list)/len(time_list) 
    return avg_inference_time
def test_model_inference_speed_cpsc_champion(loader, model, device):
    
    for net in model:
        net.eval()
    time_list=[]
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(loader)):
            if(args.dataset_name=='shaoxing-ninbo'):
                data=torch.nan_to_num(data)
            data = data.to(device)            
            start_time = time.time()
            for i in range(len(model)):            
                output=model[i](data)
            end_time = time.time()
            time_list.append(end_time-start_time)
    avg_inference_time = sum(time_list)/len(time_list) 
    return avg_inference_time


def write_inference_time_csv(args,model_name, inference_time):
    csv_path=f'/media/user/nvme0/ECG-modeling/{args.dataset_name}/inference_time.csv'
    dicts={"model_name":model_name,
           "use_gpu":args.use_gpu,
           "inference_time":inference_time
           }
    df = pd.DataFrame(dicts)
    if(os.path.isfile(csv_path)):
        df.to_csv(csv_path, mode='a', index=None, header=None)
    else:
        df.to_csv(csv_path,index=None)

class Args:
    dataset_name='cpsc_2018'
    leads = 'all'
    seed = 42    
    batch_size = 32
    num_workers = 8
    phase = 'train'#'train'
    
    resume = True
    use_gpu = False  # Set to True if you want to use GPU and it's available
    
    model_used='resnet34'
    lr = 0.0001
    if model_used=='MLBF_net':
        lr=0.001 
    elif model_used =='Transformer':
        lr=0.001
    else:
        lr=0.0001
    
    #lr=0.001
    
    threshold=0.5
    model_precision = 32
    #'lstm', 'resnet18', 'resnet34', 'GRU',
    # 'retnet','Mamba'
    # 'Residual_Conv_GRU','Residual_Conv_LSTM',Residual_ConvTransformer,Residual_conv_retnet, Residual_Conv_Mamba
    # 'mini_Residual_Conv_GRU','mini_Residual_ConvTransformer'
    
    # baseline models
    # CLINet, ResU_Dense, MLBF_net, cpsc_champion, SGB

    data_dir = f'/media/user/nvme0/ECG-modeling/{dataset_name}/datas'    
    model_dir = f'/media/user/nvme0/ECG-modeling/{dataset_name}/models/'
    if(dataset_name=='cpsc_2018'):  
        num_classes = 9  # Set this to your number of classes
    elif(dataset_name=='ptb-xl'):
        num_classes = 5  # Set this to your number of classes
    elif(dataset_name=='shaoxing-ninbo'):
        num_classes = 63  # Set this to your number of  
    if resume:
        resume_model_path=model_dir+"resnet34_2024-09-08-19:02_all_42_prune16"
    
    gpu_number=0
    quantize=False
    prune = True

args = Args()

if __name__ == "__main__":
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    folds = split_data(seed=args.seed)   
    test_folds = folds[9:]    
    train_val_folds = folds[:9]
    label_csv = os.path.join(data_dir, 'labels.csv')
    leads='all'   

    if(args.dataset_name == 'cpsc_2018'):
        print("CPSC-2018")
        test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    elif(args.dataset_name == 'ptb-xl'):
        print("PTB-xl dataset")
        test_dataset = PTBDataset('test', data_dir, label_csv, test_folds, leads)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    elif(args.dataset_name == 'shaoxing-ninbo'):
        print("shaoxing_nimbo dataset")
        test_dataset = shaoxingDataset('test', data_dir, label_csv, test_folds, leads)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        print("Unsupported dataset")

    device = torch.device(f'cuda:{args.gpu_number}') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    print(device)
    inference_time_list=[]
    model_name_list=[]

    # with open(f"/media/user/nvme0/ECG-modeling/configuration/model_path.txt") as f:
    with open(f"/media/user/nvme0/ECG-modeling/configuration/model_path_prune.txt") as f:
        lines=f.read().splitlines()
        for line in lines:
            
            if(args.model_used=='cpsc_champion'):
                path=f'{args.model_dir}{line}_fold0_ensemble.pth'
                net =[cpsc_champion(seq_len=15000, num_classes=args.num_classes) for i in range(13)]
                for i in range(13):
                    net[i] = net[i].to(device)
                for i in range(13):
                    if(i<12):
                        if(args.dataset_name=='cpsc_2018'):
                            name ="cpsc_champion_2024-08-10-16:04_all_42"
                        elif(args.dataset_name=='ptb-xl'):                            
                            name="cpsc_champion_2024-08-10-16:20_all_42"
                        else:
                            name="cpsc_champion_2024-08-10-16:18_all_42"
                        path = args.model_dir+name+f'_lead{i}_fold0.pth'
                    else:
                        if(args.dataset_name=='cpsc_2018'):
                            name ="cpsc_champion_2024-07-31-23:08_all_42"
                        elif(args.dataset_name=='ptb-xl'):                            
                            name="cpsc_champion_2024-07-31-23:08_all_42"
                        else:
                            name="cpsc_champion_2024-08-01-19:02_all_42"
                        
                        path = args.model_dir+name+f'_fold0.pth'
                    args.model_path=path                    
                    net[i].load_state_dict(torch.load(args.model_path, map_location=device))

                inference_time = test_model_inference_speed_cpsc_champion(test_loader,net,device)
                print(inference_time)
                inference_time_list.append(inference_time)
                break
            elif(args.quantize):
                print(line)
                tmp=line.split('_2')
                args.model_used=tmp[0]
                print(args.model_used)
                path=f'{args.model_dir}{line}_fold0.pth'        
                model_name_list.append(path)
                net = model_initialize(12, args, device)                
                net.load_state_dict(torch.load(path, map_location=device))
                net=net.half()
                inference_time = test_model_inference_speed(args,test_loader,net,device)
                print(inference_time)
                inference_time_list.append(inference_time)
            elif(args.prune):                
                tmp=line.split('_2')
                args.model_used=tmp[0]
                path=f'{args.model_dir}{line}_fold0.pth'     
                print(path)
                model_name_list.append(path)
                net = torch.load(path,map_location=device)
                # net = model_initialize(12, args, device)                
                # net.load_state_dict(torch.load(path, map_location=device))
                inference_time = test_model_inference_speed(args,test_loader,net,device)
                print(inference_time)
                inference_time_list.append(inference_time)

            else:
                print(line)
                tmp=line.split('_2')
                args.model_used=tmp[0]
                print(args.model_used)
                path=f'{args.model_dir}{line}_fold0.pth'        
                model_name_list.append(path)
                net = model_initialize(12, args, device)
                #print(net)
                net.load_state_dict(torch.load(path, map_location=device))
                inference_time = test_model_inference_speed(args,test_loader,net,device)
                print(inference_time)
                inference_time_list.append(inference_time)

    write_inference_time_csv(args,model_name_list,inference_time_list)







