import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import ECGDataset,PTBDataset,shaoxingDataset
from utils.utils import split_data
import datetime
import time
from utils.models.models import Residual_Conv_GRU
import numpy as np
# from threading import Thread
from torch.multiprocessing import Process, Queue
import warnings

from tqdm import tqdm
import pandas as pd

#import torch.optim as optim
#from torch.optim.lr_scheduler import StepLR


def train_test(trainloader, testloader, model, dataset_name, criterion, num_epochs, optimizer, device, queue, 
               model_name):
    model.to(device)
    model.train()
    if(dataset_name=='cpsc_2018'):  
        num_classes = 9  
    elif(dataset_name=='ptb-xl'):
        num_classes = 5      
    elif(dataset_name=='shaoxing-ninbo'):
        num_classes = 63 

    train_acc_epoch = []
    test_acc_epoch = []
    for epoch in range(num_epochs):
        model.train()
        sample_per_label=torch.zeros(num_classes).to(device)
        accuracy_per_label = []
        running_loss = 0
        correct_preds_per_label = None
        total_preds_per_label = None
        print("epoch : ", epoch)        
        for n, (data, labels) in enumerate(trainloader):        
            data, labels = data.to(device), labels.to(device)
            if(dataset_name=='shaoxing-ninbo'):
                data=torch.nan_to_num(data)
                
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted = torch.sigmoid(outputs).data > 0.5
            running_loss += loss.item()

            # Update correct and total predictions for each label
            if correct_preds_per_label is None:
                correct_preds_per_label = (predicted == labels).sum(axis=0)
                total_preds_per_label = labels.size(0)
            else:
                correct_preds_per_label += (predicted == labels).sum(axis=0)
                total_preds_per_label += labels.size(0)    
            for label in labels:
                sample_per_label+=label
                
        #scheduler.step()
        avg_loss = running_loss / len(trainloader)   
        print(f'loss : {avg_loss}')
        for i in range(labels.size(1)):
            accuracy = correct_preds_per_label[i].item() / total_preds_per_label
            accuracy_per_label.append(accuracy)

        num_samples=sample_per_label.tolist()
        tmp = np.array(accuracy_per_label) * num_samples/sum(num_samples)
        balanced_accuracy=sum(tmp)
        print(f"{dataset_name}:balanced_accuracy : ", balanced_accuracy)        
        train_acc= balanced_accuracy
        # evaluate 
        model_path=f"/mnt/8tb_raid/ECG-modeling/{dataset_name}/models/{model_name}.pth"
        test_acc = evaluate(testloader,model,dataset_name,device,model_path)
        train_acc_epoch.append(train_acc)
        test_acc_epoch.append(test_acc)

    queue.put((train_acc,test_acc,train_acc_epoch, test_acc_epoch))

def evaluate(dataloader, model, dataset_name, device,model_path):
    model.to(device)
    model.eval()

    if(dataset_name=='cpsc_2018'):  
        num_classes = 9  
    elif(dataset_name=='ptb-xl'):
        num_classes = 5      
    elif(dataset_name=='shaoxing-ninbo'):
        num_classes = 63 

    sample_per_label=torch.zeros(num_classes).to(device)
    accuracy_per_label = []
    correct_preds_per_label = None
    total_preds_per_label = None

    with torch.no_grad():
        for ii, (data, labels) in enumerate(tqdm(dataloader)):           
            data, labels = data.to(device), labels.to(device)            
            if(dataset_name=='shaoxing-ninbo'):
                data=torch.nan_to_num(data)
            
            output = model(data)
            output = torch.sigmoid(output)
            predicted = output.data > 0.5
            
            # Update correct and total predictions for each label
            if correct_preds_per_label is None:
                correct_preds_per_label = (predicted == labels).sum(axis=0)                
                total_preds_per_label = labels.size(0)
            else:
                correct_preds_per_label += (predicted == labels).sum(axis=0)
                total_preds_per_label += labels.size(0)  
            for label in labels:
                sample_per_label+=label

        num_samples=sample_per_label.tolist()
        for i in range(labels.size(1)):
            accuracy = correct_preds_per_label[i].item() / total_preds_per_label
            accuracy_per_label.append(accuracy)

        tmp = np.array(accuracy_per_label) * num_samples/sum(num_samples)
        balanced_accuracy=sum(tmp)
        print(f"{dataset_name}_test:balanced_accuracy : ", balanced_accuracy)
        #save model
        
        torch.save(model.state_dict(), model_path)

    return balanced_accuracy

def average_model_params_without__make_layer(model1, model2, model3):  #FTL with only feature extraction layer
    # Get state dictionaries (parameter values) of models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    state_dict3 = model3.state_dict()    
    # Average the parameters from only feature extration layers
    averaged_state_dict = {}
    for key in state_dict1:
        # Skip averaging for the fully connected layer
        
        if "resnet_half.0" in key or "resnet_half.1" in key:
            averaged_state_dict[key] = (state_dict1[key] + state_dict2[key]+state_dict3[key]) / 3
        else:
            continue
    
    return averaged_state_dict

def average_model_params_without_gru(model1, model2, model3): #FTL without classification block
    # Get state dictionaries (parameter values) of models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    state_dict3 = model3.state_dict()    
    
    averaged_state_dict = {}
    for key in state_dict1:
        # Skip averaging for the fully connected layer and GRU
        
        if "fc" in key or "gru" in key:
            continue
        else:
            averaged_state_dict[key] = (state_dict1[key] + state_dict2[key]+state_dict3[key]) / 3
    
    return averaged_state_dict

def average_model_params(model1, model2, model3): # FTL with all blocks
    # Get state dictionaries (parameter values) of models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    state_dict3 = model3.state_dict()    
    # Average the parameters from models, excluding the fully connected layer
    averaged_state_dict = {}
    for key in state_dict1:
        # Skip averaging for the fully connected layer
        if "fc" in key:
            continue
        else:
            averaged_state_dict[key] = (state_dict1[key] + state_dict2[key]+state_dict3[key]) / 3

    return averaged_state_dict

def write_models_acc_each_epoch(acc0,acc1,acc2, epochs,phase,iteration, model_name):
    csv_path=f"/mnt/8tb_raid/ECG-modeling/foundation_model_each_epoch_{phase}.csv"
    
    avg_acc=np.zeros(epochs)
    for i in range(len(acc0)):
        avg_acc[i]=(acc0[i]+acc1[i]+acc2[i])/3

    dicts={'name':[model_name for i in range(epochs)],           
           'iter':[iteration for i in range(epochs)],
           'epoch':[i for i in range(epochs)], 
           'cpsc_2018':acc0, 
           'ptb-xl':acc1,  
           'shaoxing-ninbo':acc2, 
           'avg':avg_acc}
    df = pd.DataFrame(dicts) 

    if(os.path.isfile(csv_path)):
        df.to_csv(csv_path, mode='a', index=None, header=None)
    else:
        df.to_csv(csv_path,index=None)

def write_models_acc_csv(acc0,acc1,acc2,phase,iteration, model_name):
    csv_path="/mnt/8tb_raid/ECG-modeling/foundation_model_summary.csv"
    col=['name','phase', 'iter','cpsc_2018', 'ptb-xl',  'shaoxing-ninbo', 'avg']
    df = pd.DataFrame(columns=col)    
    avg_acc=(acc0+acc1+acc2)/3
    df.loc[0]= [model_name, phase, iteration,str(acc0), str(acc1), str(acc2),str(avg_acc)]

    if(os.path.isfile(csv_path)):
        df.to_csv(csv_path, mode='a', index=None, header=None)
    else:
        df.to_csv(csv_path,index=None)

def get_data_loader(datasets, label_csv_list, data_dir_list, args,  leads='all', normalize=False):
    train_loaders = []
    test_loaders = []
    folds = split_data(seed=args.seed)   
    test_folds = folds[9:]    
    train_folds = folds[:9] 
    

    for dataset_name, label_csv, data_dir in zip(datasets, label_csv_list, data_dir_list):
        if(dataset_name=='cpsc_2018'):
            train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads, normalize)
            test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads, normalize)
        elif(dataset_name=='ptb-xl'):
            train_dataset = PTBDataset('train', data_dir, label_csv, train_folds, leads, normalize)
            test_dataset = PTBDataset('test', data_dir, label_csv, test_folds, leads, normalize)
        elif(dataset_name=='shaoxing-ninbo'):
            train_dataset = shaoxingDataset('train', data_dir, label_csv, train_folds, leads, normalize)
            test_dataset = shaoxingDataset('test', data_dir, label_csv, test_folds, leads, normalize)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return  train_loaders, test_loaders

class Args:
    
    leads = 'all'
    seed = 42    
    batch_size = 32
    num_workers = 8
    phase = 'train'
    epochs = 10
    folds = 10
    resume = False
    use_gpu = True  # Set to True if you want to use GPU and it's available
    ct = str(datetime.datetime.now())[:10]+'-'+str(datetime.datetime.now())[11:16]
    model_used='Residual_Conv_GRU'
    lr = 0.0001
    if model_used=='MLBF_net':
        lr=0.001 
    elif model_used =='Transformer':
        lr=0.001
    else:
        lr=0.0001
    
    
    model_precision = 32
    #'lstm', 'resnet18', 'resnet34', 'GRU',
    # 'retnet','Mamba'
    # 'Residual_Conv_GRU','Residual_Conv_LSTM',Residual_ConvTransformer,Residual_conv_retnet, Residual_Conv_Mamba
    # 'mini_Residual_Conv_GRU','mini_Residual_ConvTransformer'    
    # baseline models
    # CLINet, ResU_Dense, MLBF_net, cpsc_champion, SGB
    model_name=f'{model_used}_{ct}'
    
    iteration = 10
    normalize= False
    
args = Args()
warnings.filterwarnings('ignore', category=FutureWarning)


if __name__ == "__main__":   

    datasets =['cpsc_2018', 'ptb-xl',  'shaoxing-ninbo']
    num_classes=[9, 5, 63]
    
    data_dir_list = [os.path.normpath(f'/mnt/8tb_raid/ECG-modeling/{dataset_name}/datas')  for dataset_name in datasets] 
    label_csv_list = [os.path.join(data_dir, 'labels.csv') for data_dir in data_dir_list]
    model_list=[]
    for dataset_num_classes in num_classes:
        model_list.append(Residual_Conv_GRU( input_size=15000, batch_size=args.batch_size, input_channels=12, 
                            num_classes=dataset_num_classes,GRU_hidden_size=128,GRU_num_layers=2))


    train_loaders, test_loaders = get_data_loader(datasets, label_csv_list, data_dir_list, args, normalize=args.normalize)

    optimizer0 = torch.optim.Adam(model_list[0].parameters(), lr=args.lr)
    optimizer1 = torch.optim.Adam(model_list[1].parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(model_list[2].parameters(), lr=args.lr)
    
   
    
    
    criterion = nn.BCEWithLogitsLoss()


    torch.multiprocessing.set_start_method('spawn', force=True)
    for  it in range(args.iteration):        
        
        queue0 = Queue()
        queue1 = Queue()
        queue2 = Queue()

        p1 = Process(target=train_test, args=(train_loaders[0], test_loaders[0], model_list[0], "cpsc_2018",
                                        criterion, args.epochs, optimizer0, 'cuda:0',queue0, args.model_name))
        p2 = Process(target=train_test, args=(train_loaders[1], test_loaders[1], model_list[1], "ptb-xl",
                                        criterion, args.epochs, optimizer1, 'cuda:1',queue1,args.model_name))
        p3 = Process(target=train_test, args=(train_loaders[2], test_loaders[2], model_list[2], "shaoxing-ninbo",
                                        criterion, args.epochs, optimizer2, 'cuda:2', queue2,args.model_name))
        
        # Start the processes
        p1.start()
        p2.start()
        p3.start()
        
        # Wait for processes to finish
        p1.join()
        p2.join()
        p3.join()
               
        train_acc0, test_acc0, train_acc_epoch0, test_acc_epoch0 = queue0.get()
        train_acc1, test_acc1, train_acc_epoch1, test_acc_epoch1 = queue1.get()
        train_acc2, test_acc2, train_acc_epoch2, test_acc_epoch2 = queue2.get()
        write_models_acc_csv(train_acc0, train_acc1, train_acc2, 'train', it, args.model_name)
        write_models_acc_csv(test_acc0, test_acc1, test_acc2, 'test', it, args.model_name)

        write_models_acc_each_epoch(train_acc_epoch0, train_acc_epoch1, train_acc_epoch2, 
                                    args.epochs, 'train', it, args.model_name)
        write_models_acc_each_epoch(test_acc_epoch0, test_acc_epoch1, test_acc_epoch2, 
                                    args.epochs, 'test', it, args.model_name)
        
        # average model param
        #avg_state_dict = average_model_params(model_list[0],model_list[1],model_list[2])#with all block

        avg_state_dict = average_model_params_without_gru(model_list[0],model_list[1],model_list[2]) #without classification block

        #avg_state_dict = average_model_params(model_list[0],model_list[1],model_list[2]) #with only feature extraction

        #reload_model_param
        model_list[0].load_state_dict(avg_state_dict, strict=False)
        model_list[1].load_state_dict(avg_state_dict, strict=False)
        model_list[2].load_state_dict(avg_state_dict, strict=False)
