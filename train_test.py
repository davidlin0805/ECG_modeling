import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import ECGDataset,PTBDataset,shaoxingDataset
from utils.utils import split_data
from utils.models.models import resnet34,resnet18,LSTMModel,GRUModel,Mamba,RetNet
from utils.models.models import Residual_Conv_GRU, Residual_Conv_LSTM, Residual_ConvTransformer,Residual_conv_retnet, Residual_Conv_Mamba
from utils.models.models import mini_Residual_Conv_GRU, mini_Residual_ConvTransformer, Transformer
from utils.models.models import ResU_Dense, MLBF_net, SGB, cpsc_champion,CLINet

import warnings
from tqdm import tqdm
import numpy as np
import datetime
import time
from argparse import ArgumentParser
from utils.utils import write_csv, write_summary_csv, print_prediction
from utils.pruning import prune


def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device, dataset):
    print(f'Training epoch {epoch}:')
    net.train()
    running_loss = 0
    correct_preds_per_label = None
    total_preds_per_label = None    
    output_list, labels_list = [], []
    accuracy_per_label = []
    sample_per_label=torch.zeros(args.num_classes).to(device)
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)        
        if(args.dataset_name=='shaoxing-ninbo'):
            data=torch.nan_to_num(data)
        if(args.quantize and args.quantization_precision==16):
            if(args.model_used=='Residual_Conv_GRU' or args.model_used=='ResU_Dense'):
                outputs = net(data.half(),quantize=True) #need to conert h0 
            else:
                outputs = net(data.half())
            loss = criterion(outputs, labels)           
        else:
            outputs = net(data)
            loss = criterion(outputs, labels)            
            loss.backward()
            optimizer.step()             

        running_loss += loss.item()
        
        # Convert outputs to binary predictions
        predicted = torch.sigmoid(outputs).data > 0.5

        # Update correct and total predictions for each label
        if correct_preds_per_label is None:
            correct_preds_per_label = (predicted == labels).sum(axis=0)
            total_preds_per_label = labels.size(0)
        else:
            correct_preds_per_label += (predicted == labels).sum(axis=0)
            total_preds_per_label += labels.size(0)

        output_list.append(predicted.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())     

        for label in labels:
            sample_per_label+=label


    num_samples=sample_per_label.tolist()
    y_pred=np.vstack(output_list)  
    y_true=np.vstack(labels_list)
    
    for i in range(labels.size(1)):
        accuracy = correct_preds_per_label[i].item() / total_preds_per_label
        accuracy_per_label.append(accuracy)
       
    tmp = np.array(accuracy_per_label) * num_samples/sum(num_samples)
    balanced_accuracy=sum(tmp)
    
    avg_accuracy,avg_precision, avg_recall, avg_f1, balanced_accuracy=cal_score(y_true,y_pred, args, 'train',dataset,balanced_accuracy, accuracy_per_label) 
    avg_loss = running_loss / len(dataloader)    

    print(f'Loss: {avg_loss:.4f}')
    print(f'Balanced Accuracy:{balanced_accuracy:.4f}')
    #print(f'Average Accuracy: {avg_accuracy:.4f}')
    #print(f'Average Precision: {avg_precision:.4f}')
    #print(f'Average Recall: {avg_recall:.4f}')
    #print(f'Average F1 Score: {avg_f1:.4f}')

    return avg_accuracy, avg_precision, avg_recall, avg_f1, balanced_accuracy


def evaluate(dataloader, net, args, criterion, device, model_path,dataset):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    correct_preds_per_label = None
    total_preds_per_label = None
    inference_time = []
    accuracy_per_label = []
    sample_per_label=torch.zeros(args.num_classes).to(device)
    with torch.no_grad():
        for _, (data, labels) in enumerate(tqdm(dataloader)):           
            data, labels = data.to(device), labels.to(device)            
            if(args.dataset_name=='shaoxing-ninbo'):
                data=torch.nan_to_num(data)
            start_time = time.time()            
            if(args.quantize and args.quantization_precision==16):
                if(args.model_used=='Residual_Conv_GRU' or args.model_used=='ResU_Dense'):
                    output = net(data.half(),quantize=True)
                else:
                    output = net(data.half())
            else:
                output = net(data)
            loss = criterion(output, labels) 
            output = torch.sigmoid(output)
            predicted = output.data > 0.5            
            end_time = time.time()
            inference_time.append((end_time - start_time))
            output_list.append(predicted.data.cpu().numpy())
            labels_list.append(labels.data.cpu().numpy())            

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

        y_pred=np.vstack(output_list)  
        y_true=np.vstack(labels_list)
            
        avg_loss = running_loss / len(dataloader)
        avg_accuracy,avg_precision, avg_recall, avg_f1, balanced_accuracy=cal_score(y_true,y_pred, args, 'test',dataset,balanced_accuracy, accuracy_per_label)
        
        avg_inference_time = sum(inference_time)/len(inference_time)

        
        #print(f'Loss: {avg_loss:.4f}')
        print(f'Balanced Accuracy:{balanced_accuracy:.4f}')
        #print(f'Average Accuracy: {avg_accuracy:.4f}')
        #print(f'Average Precision: {avg_precision:.4f}')
        #print(f'Average Recall: {avg_recall:.4f}')
        #print(f'Average F1 Score: {avg_f1:.4f}')
        print(f'Average inference speed:{avg_inference_time:.4f}')

        if args.phase == 'train' and balanced_accuracy>args.best_metric:
            args.best_metric = balanced_accuracy
            print(model_path)
            if(args.prune):
                torch.save(net, model_path) # without .state_dict
            else:
                torch.save(net.state_dict(), model_path)
                

    return avg_accuracy, avg_precision, avg_recall, avg_f1, avg_inference_time, balanced_accuracy

def cal_score(y_true,y_pred, args, phase,dataset, balanced_accuracy, accuracy_per_label): 
    #cal accuracy and write per label results
    precision_per_label=[]
    recall_per_label=[]
    f1_per_label=[]

    cm=np.zeros((args.num_classes,args.num_classes))
    for i,j in zip(y_true,y_pred):
        nonzeroi=np.nonzero(i)[0]        
        nonzeroj=np.nonzero(j)[0]
        for idxi in nonzeroi:
            for idxj in nonzeroj:
                cm[idxi][idxj]+=1

    for c in range(args.num_classes):
        tp = cm[c,c]
        fp = sum(cm[:,c]) - cm[c,c]
        fn = sum(cm[c,:]) - cm[c,c]
        tn = sum(np.delete(sum(cm)-cm[c,:],c))
        recall = tp/(tp+fn) if(tp+fn!=0) else 0
        precision = tp/(tp+fp) if(tp+fp!=0) else 0
        f1_score = 2*((precision*recall)/(precision+recall)) if(precision+recall!=0) else 0
        precision_per_label.append(precision)
        recall_per_label.append(recall)
        f1_per_label.append(f1_score)
    
    avg_accuracy = sum(accuracy_per_label) / len(accuracy_per_label)
    avg_precision = sum(precision_per_label) / len(precision_per_label)
    avg_recall = sum(recall_per_label) / len(recall_per_label)
    avg_f1 = sum(f1_per_label) / len(f1_per_label)
    
    if(args.phase=="test"):
        csv_file_name=f"{args.model_path[:-4]}.csv"
    elif(args.quantize):
        csv_file_name=f"{args.model_path[:-4]}_quantize.csv"
    elif(args.prune):
        csv_file_name=f"{args.model_path[:-4]}_prune.csv"
    else:
        csv_file_name=f'{args.model_dir[:-7]}/results/{args.model_name}_{phase}_{args.train_split}%.csv'
    write_csv(csv_file_name, dataset, accuracy_per_label, precision_per_label, recall_per_label, f1_per_label, 
        avg_accuracy, avg_precision, avg_recall, avg_f1) 

    return avg_accuracy,avg_precision, avg_recall, avg_f1, balanced_accuracy

def model_initialize(nleads,args,device):
    # Initialize model
    if(args.model_used=='lstm'):
        input_size = 15000  # Number of ECG leads
        hidden_size = 128  # Adjust as needed
        num_layers = 2  # Adjust as needed
        net = LSTMModel(input_size, hidden_size, num_layers, args.num_classes)
        net = net.to(device)
    elif(args.model_used=='resnet18'):
        net = resnet18(input_channels=nleads, num_classes=args.num_classes).to(device)
    elif(args.model_used=='resnet34'):
        net = resnet34(input_channels=nleads, num_classes=args.num_classes).to(device)
    elif(args.model_used=='GRU'):
        input_size = 15000  # Number of ECG leads
        hidden_size = 128  # Adjust as needed
        num_layers = 2  # Adjust as needed
        net = GRUModel(input_size, hidden_size, num_layers, args.num_classes)
        net = net.to(device)
    elif(args.model_used=='retnet'):
        print("retnet")
        hidden_size=32
        ffn_size=32
        sequence_len=15000
        features = 12
        net = RetNet(4, hidden_size, ffn_size, heads=4, sequence_length=sequence_len, features=features, num_classes=args.num_classes, double_v_dim=False)
        net = net.to(device)
    elif(args.model_used=='Mamba'):
        print("mamba")        
        d_model = 32 # dimension of model
        expand = 2
        enc_in = 2 
        c_out = args.num_classes           
        net = Mamba(d_model, expand, enc_in, c_out, d_conv=4, d_ff=32, e_layers=2, dropout=0.05)
        net = net.to(device)
    elif(args.model_used=='Residual_Conv_GRU' or args.model_used=='mini_Residual_Conv_GRU'):
        print(args.model_used)
        input_size=15000
        hidden_size = 128  # Adjust as needed
        num_layers = 2  # Adjust as needed
        if(args.model_used=='Residual_Conv_GRU'):
            net = Residual_Conv_GRU( input_size=input_size, batch_size=args.batch_size, input_channels=nleads, 
                                num_classes=args.num_classes,GRU_hidden_size=hidden_size,GRU_num_layers=num_layers)
        else:
            net = mini_Residual_Conv_GRU( input_size=input_size, batch_size=args.batch_size, input_channels=nleads, 
                                num_classes=args.num_classes,GRU_hidden_size=hidden_size,GRU_num_layers=num_layers)
        net = net.to(device)
    elif(args.model_used=='Residual_Conv_LSTM'):
        print('Residual_Conv_LSTM')
        input_size=15000
        hidden_size = 128  # Adjust as needed
        num_layers = 2  # Adjust as needed
        net = Residual_Conv_LSTM( input_size=input_size, batch_size=args.batch_size, input_channels=nleads, 
                                num_classes=args.num_classes, hidden_size=hidden_size, num_layers=num_layers)
        net = net.to(device)
    elif(args.model_used=='Residual_ConvTransformer' or args.model_used=='mini_Residual_ConvTransformer'):
        print(args.model_used)
        input_size=15000
        if(args.model_used=='Residual_ConvTransformer'):
            net = Residual_ConvTransformer( input_size=input_size, batch_size=args.batch_size, input_channels=nleads, 
                                num_classes=args.num_classes, num_layers=4)
        else:
            net = mini_Residual_ConvTransformer( input_size=input_size, batch_size=args.batch_size, input_channels=nleads, 
                                num_classes=args.num_classes, num_layers=2)
        net = net.to(device)
        
    elif(args.model_used=='Residual_conv_retnet'):
        
        input_size=15000
        net = Residual_conv_retnet( input_size=input_size, batch_size=args.batch_size, input_channels=nleads, 
                                num_classes=args.num_classes, num_layers=2,hidden_dim=64,ffn_size=128)
        net = net.to(device)
    elif(args.model_used == 'CLINet'):
        net = CLINet(sequence_len=15000, num_features=nleads,  num_classes=args.num_classes) 
        net = net.to(device)
    elif(args.model_used=='ResU_Dense'):
        net = ResU_Dense(nOUT = args.num_classes, in_ch = 12, out_ch = 256, mid_ch = 64)
        net = net.to(device)
    elif(args.model_used=='MLBF_net'):
        net = MLBF_net(nleads=nleads, num_classes=args.num_classes)
        net = net.to(device)
    elif(args.model_used=='SGB'):
        net = SGB(num_classes=args.num_classes)
        net = net.to(device)
    elif(args.model_used=='cpsc_champion'):
        net = cpsc_champion(seq_len=15000, num_classes=args.num_classes)
        net = net.to(device)
    elif(args.model_used=='Residual_Conv_Mamba'):
        net = Residual_Conv_Mamba(d_model=32, expand=2, d_conv=8, d_ff=32, e_layers=2, num_classes=args.num_classes)
        net = net.to(device)
    elif(args.model_used=='Transformer'):
        net = Transformer(num_classes=args.num_classes, num_layers=2, model_dim=64)
        net = net.to(device)
    
    return net


warnings.filterwarnings('ignore', category=FutureWarning)

class Args:
    dataset_name='cpsc_2018'

    leads = 'all'
    seed = 42
    
    lr = 0.0001
    batch_size = 32
    num_workers = 4
    phase = 'test'
    epochs = 1
    folds = 10
    train_split = 9 #   train_split*10 = percentage

    resume = False
    use_gpu = True  # Set to True if you want to use GPU and it's available
    ct = str(datetime.datetime.now())[:10]+'-'+str(datetime.datetime.now())[11:16]
    
    model_used='Residual_Conv_GRU'
    model_precision = 32

    #'lstm', 'resnet18', 'resnet34', 'GRU',
    # 'retnet','Mamba'
    # 'Residual_Conv_GRU','Residual_Conv_LSTM',Residual_ConvTransformer,Residual_conv_retnet, Residual_Conv_Mamba
    # 'mini_Residual_Conv_GRU','mini_Residual_ConvTransformer'
    
    # baseline models
    # CLINet, ResU_Dense, MLBF_net, cpsc_champion, SGB

    
    data_dir = f'/mnt/8tb_raid/ECG-modeling/{dataset_name}/datas/'
    
    model_dir = f'/mnt/8tb_raid/ECG-modeling/{dataset_name}/models/'

    if(dataset_name=='cpsc_2018'):
        num_classes = 9  # Set this to your number of classes
        
        if(phase=='train'):
            model_name=f'{model_used}_{ct}_{leads}_{seed}'
            model_path =model_dir+model_name
    elif(dataset_name=='ptb-xl'):
        num_classes = 5  # Set this to your number of classes
        if(phase=='train'):
            model_name=f'{model_used}_{ct}_{leads}_{seed}'
            model_path =model_dir+model_name
        
    elif(dataset_name=='shaoxing-ninbo'):
        num_classes = 63  # Set this to your number of classes
        if(phase=='train'):
            model_name=f'{model_used}_{ct}_{leads}_{seed}'
            model_path =model_dir+model_name

        
    if(resume):
        model_path="/mnt/8tb_raid/ECG-modeling/cpsc_2018/models/SGB_2024-07-30-16:22_all_42_fold8.pth"
    


    quantize = False
    prune = False
    
    if quantize or prune:
        phase="test_quantize_or_pruning"
    

    
    quantization_precision = 16
    if phase=="test":
        model_path = model_dir+'Residual_Conv_GRU_2024-11-27-18:28_all_42_foundation_model_unfreeze_True.pth'
    elif quantize :
        model_path=model_dir+"Residual_Conv_GRU_2024-11-27-18:28_all_42_foundation_model_unfreeze_True.pth"    
    elif prune :
        model_path = model_dir+'Residual_Conv_GRU_2024-11-27-18:28_all_42_foundation_model_unfreeze_True.pth'
   
    
    gpu_number=2

args = Args()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-config_file", help="config_file",dest="config_file",default=None)
    arguments = parser.parse_args()
    if(arguments.config_file):
        with open(arguments.config_file,'r') as f:
            lines=f.read().splitlines()
            for line in lines:
                tmp=line.split(':')
                if(tmp[0]=='lr'):
                    val=float(tmp[1]) 
                else:
                    val=int(tmp[1]) if tmp[1].isnumeric()  else tmp[1]
                setattr(args,tmp[0],val)
        args.model_dir = f'/mnt/8tb_raid/ECG-modeling/{args.dataset_name}/models/'
        args.data_dir = f'/mnt/8tb_raid/ECG-modeling/{args.dataset_name}/datas/'  
        args.model_name = f'{args.model_used}_{args.ct}_{args.leads}_{args.seed}'
        args.model_path = args.model_dir + args.model_name
    print(args.dataset_name)

    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)    

    # Set the model path if it's not already set
    if not args.model_path:
        args.model_path = f'/mnt/8tb_raid/ECG-modeling/ptb-xl/models/lstm_{database}_{args.leads}_{args.seed}.pth'
    
    # Ensure the 'models' directory exists
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if(args.quantize):
        #gpu_available = False
        #device = torch.device('cpu')
        gpu_available=torch.cuda.is_available()
        device = torch.device(f'cuda:{args.gpu_number}') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    else:
        gpu_available=torch.cuda.is_available()
        print("use_gpu : ",gpu_available)    
        device = torch.device(f'cuda:{args.gpu_number}') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    
    leads = args.leads.split(',') if args.leads != 'all' else 'all'
    nleads = len(leads) if args.leads != 'all' else 12
    
    label_csv = os.path.join(data_dir, 'labels.csv')       
    folds = split_data(seed=args.seed)  
    test_folds = folds[args.train_split:]    
    train_folds = folds[:args.train_split]    
    args.best_metric = 0
    # Initialize model
    
    net = model_initialize(nleads, args,device)
    print(args.model_used)
    if(args.prune):
        print(args.model_path)
        path = args.model_path
        print(path)
        pruned_model = prune(args, path, device, net)
        net = pruned_model.to(device)
        net.eval()
    elif(args.quantize):
        path = args.model_path
        print(path)
        net.load_state_dict(torch.load(path, map_location=device))          
        net.eval()
        quantized_model=net.half()
        net = quantized_model.to(device)
        net.eval()        
    
    

    size_model=0
    for param in net.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    model_size = size_model/ 1e6/8 #MB
    print(model_size)
    

    if(args.dataset_name == 'cpsc_2018'):
        print("CPSC-2018")
        train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    elif(args.dataset_name == 'ptb-xl'):
        print("PTB-xl dataset")
        train_dataset = PTBDataset('train', data_dir, label_csv, train_folds, leads)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        test_dataset = PTBDataset('test', data_dir, label_csv, test_folds, leads)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    elif(args.dataset_name == 'shaoxing-ninbo'):
        print("shaoxing_ninbo dataset")
        train_dataset = shaoxingDataset('train', data_dir, label_csv, train_folds, leads)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        test_dataset = shaoxingDataset('test', data_dir, label_csv, test_folds, leads)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        print("Unsupported dataset")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()  # or another appropriate loss function
    
    if args.phase == 'train' or args.phase == 'test_quantize_or_pruning':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))        
        
        for epoch in range(args.epochs):                
            (avg_accuracy, avg_precision, avg_recall, avg_f1,balanced_accuracy)=train(train_loader, net,
                                                                            args, criterion, epoch, 
                                                                            scheduler, optimizer, device, train_dataset)
        if (args.quantize):
            model_path=args.model_path[:-4]+f"_quantize{args.quantization_precision}.pth"
        elif(args.prune):
            model_path=args.model_path[:-4]+f"prune{args.quantization_precision}.pth"
        else:
            model_path=args.model_path+f'_train_{args.train_split*10}%.pth' 

        summary_file_name=f'{args.model_dir[:-7]}/summary{args.gpu_number}.csv'
        write_summary_csv(summary_file_name, args, avg_accuracy, 
                            avg_precision, avg_recall, avg_f1, model_path, 'train',
                            gpu_available, 0, model_size,balanced_accuracy)
        
        (avg_accuracy, avg_precision, avg_recall, 
         avg_f1, avg_inference_time,balanced_accuracy)=evaluate(test_loader, 
                                                                net, args, criterion, 
                                                                device, model_path, test_dataset)        
        summary_file_name=f'{args.model_dir[:-7]}/summary{args.gpu_number}.csv'
        write_summary_csv(summary_file_name, args, avg_accuracy, 
                            avg_precision, avg_recall, avg_f1, model_path, 'test',
                            gpu_available, avg_inference_time, model_size, balanced_accuracy)    
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        (avg_accuracy, avg_precision, avg_recall, avg_f1, avg_inference_time,balanced_accuracy)=evaluate(test_loader,
                                                                                    net, args, criterion, 
                                                                                    device, args.model_path, test_dataset)
        summary_file_name=f'{args.model_dir[:-7]}/summary{args.gpu_number}.csv'
        write_summary_csv(summary_file_name, args, avg_accuracy, 
                            avg_precision, avg_recall, avg_f1, args.model_path, args.folds,
                            gpu_available, avg_inference_time, model_size, balanced_accuracy)      
           