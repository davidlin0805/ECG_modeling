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
    """
    Trains the neural network model for one epoch using the provided data loader.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader containing the training data and labels.
        net (torch.nn.Module): The neural network model to be trained.
        args (Namespace): Arguments containing training parameters such as quantization settings, model type, and dataset name.
        criterion (torch.nn.Module): The loss function to compute training loss.
        epoch (int): The current epoch number.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler to adjust the learning rate.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        device (torch.device): The device (CPU/GPU) to perform training on.
        dataset (Dataset): The dataset being used for training.

    Returns:
        tuple: A tuple containing:
            - avg_accuracy (float): The average accuracy over all samples and labels.
            - avg_precision (float): The average precision over all samples and labels.
            - avg_recall (float): The average recall over all samples and labels.
            - avg_f1 (float): The average F1 score over all samples and labels.
            - balanced_accuracy (float): The balanced accuracy considering all classes.

    Description:
        This function runs one training epoch. It processes batches of data, computes the loss, updates the model's parameters, and tracks accuracy, precision, recall, and F1 score. It also handles optional quantization and provides performance metrics for each epoch.
    """

    print(f'Training epoch {epoch}:')
    net.train()
    running_loss = 0
    correct_preds_per_label = None
    total_preds_per_label = None    
    output_list, labels_list = [], []
    accuracy_per_label = []
    sample_per_label=torch.zeros(args.num_classes).to(device)
    output_value_list=[]
    
    for n, (data, labels) in enumerate(tqdm(dataloader)):        
        data, labels = data.to(device), labels.to(device)        
        if(args.dataset_name=='shaoxing-ninbo'):
            data=torch.nan_to_num(data)
        optimizer.zero_grad()        
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
        predicted = torch.sigmoid(outputs).data > args.threshold

        # Update correct and total predictions for each label
        if correct_preds_per_label is None:
            correct_preds_per_label = (predicted == labels).sum(axis=0)
            total_preds_per_label = labels.size(0)
        else:
            correct_preds_per_label += (predicted == labels).sum(axis=0)
            total_preds_per_label += labels.size(0)

        output_list.append(predicted.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())     
        output_value_list.append(torch.sigmoid(outputs).data.cpu().numpy())
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
    print(f'Average Accuracy: {avg_accuracy:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')
    
    return avg_accuracy, avg_precision, avg_recall, avg_f1, balanced_accuracy

def evaluate(dataloader, net, args, criterion, device, model_path,dataset):
    """
    Evaluates the performance of the neural network model on the validation data.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader containing the validation data and labels.
        net (torch.nn.Module): The neural network model to be evaluated.
        args (Namespace): Arguments containing evaluation parameters such as quantization settings, model type, and dataset name.
        criterion (torch.nn.Module): The loss function to compute validation loss.
        device (torch.device): The device (CPU/GPU) to perform validation on.
        model_path (str): Path to save the model checkpoint if the validation performance improves.
        dataset (Dataset): The dataset being used for validation.

    Returns:
        tuple: A tuple containing:
            - avg_accuracy (float): The average accuracy over all samples and labels.
            - avg_precision (float): The average precision over all samples and labels.
            - avg_recall (float): The average recall over all samples and labels.
            - avg_f1 (float): The average F1 score over all samples and labels.
            - avg_inference_time (float): The average inference time for each batch.
            - balanced_accuracy (float): The balanced accuracy considering all classes.

    Description:
        This function evaluates the model using the validation dataset. It computes performance metrics such as loss, accuracy, precision, recall, F1 score, and balanced accuracy. It also tracks inference speed and saves the model if performance exceeds previous best metrics.
    """

    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    correct_preds_per_label = None
    total_preds_per_label = None
    inference_time = []
    accuracy_per_label = []
    sample_per_label=torch.zeros(args.num_classes).to(device)
    output_value_list=[]

    with torch.no_grad():
        for ii, (data, labels) in enumerate(tqdm(dataloader)):           
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

            running_loss += loss.item()
            output = torch.sigmoid(output)
            predicted = output.data > args.threshold                          
            end_time = time.time()
            inference_time.append((end_time - start_time))
            output_list.append(predicted.data.cpu().numpy())
            labels_list.append(labels.data.cpu().numpy())
            output_value_list.append(output.data.cpu().numpy())

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
        y_output=np.vstack(output_value_list)
            
        avg_loss = running_loss / len(dataloader)
        avg_accuracy,avg_precision, avg_recall, avg_f1, balanced_accuracy=cal_score(y_true,y_pred, args, 'val',dataset,balanced_accuracy, accuracy_per_label)
        
        avg_inference_time = sum(inference_time)/len(inference_time)
        
        print(f'Loss: {avg_loss:.4f}')
        print(f'Balanced Accuracy:{balanced_accuracy:.4f}')
        print(f'Average Accuracy: {avg_accuracy:.4f}')
        print(f'Average Precision: {avg_precision:.4f}')
        print(f'Average Recall: {avg_recall:.4f}')
        print(f'Average F1 Score: {avg_f1:.4f}')
        print(f'Average inference speed:{avg_inference_time:.4f}')

        if args.phase == 'train' and balanced_accuracy>args.best_metric:
            args.best_metric = balanced_accuracy
            print(model_path)
            if(args.prune):
                torch.save(net, model_path) # without .state_dict
            else:
                torch.save(net.state_dict(), model_path)
        
    if(args.quantize):
        print_prediction(args, y_pred, y_true,y_output,dataset, path=f'{args.model_dir[:-7]}/results/preds/{args.model_name}_quantize{args.quantization_precision}_fold{fold_idx}_val_preds.csv')
    else:
        print_prediction(args, y_pred, y_true,y_output,dataset, path=f'{args.model_dir[:-7]}/results/preds/{args.model_name}_fold{fold_idx}_val_preds.csv')

    return avg_accuracy, avg_precision, avg_recall, avg_f1, avg_inference_time, balanced_accuracy


def cal_score(y_true,y_pred, args, phase,dataset, balanced_accuracy, accuracy_per_label): 
    """
    Calculates performance metrics such as accuracy, precision, recall, and F1 score for each class and overall.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        args (Namespace): Arguments containing evaluation parameters.
        phase (str): Phase of the evaluation (e.g., 'train', 'val').
        dataset (Dataset): The dataset being evaluated.
        balanced_accuracy (float): The balanced accuracy computed previously.
        accuracy_per_label (list): The accuracy for each label.

    Returns:
        tuple: A tuple containing:
            - avg_accuracy (float): The average accuracy across all classes.
            - avg_precision (float): The average precision across all classes.
            - avg_recall (float): The average recall across all classes.
            - avg_f1 (float): The average F1 score across all classes.
            - balanced_accuracy (float): The calculated balanced accuracy.

    Description:
        This function computes and returns various performance metrics, including confusion matrix-based precision, recall, and F1 score for each label. It also saves these results in a CSV file for further analysis.
    """
    
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
        
    csv_file_name=f'{args.model_dir[:-7]}/results/{args.model_name}_{phase}_{fold_idx}_folds.csv'
    write_csv(csv_file_name, dataset, accuracy_per_label, precision_per_label, recall_per_label, f1_per_label, 
        avg_accuracy, avg_precision, avg_recall, avg_f1) 

    return avg_accuracy,avg_precision, avg_recall, avg_f1, balanced_accuracy

def model_initialize(nleads,args,device):
    """
    Initializes and returns a neural network model based on the specified architecture in the arguments.

    Args:
        nleads (int): The number of input channels/leads for the model.
        args (Namespace): Arguments containing model configuration (e.g., model type, number of classes).
        device (torch.device): The device (CPU/GPU) on which the model will be loaded.

    Returns:
        torch.nn.Module: The initialized neural network model.

    Description:
        Based on the `args.model_used` parameter, this function initializes various types of models such as LSTM, ResNet, GRU, RetNet, and others. The model is loaded onto the specified device (CPU or GPU).
    """
    
    print(args.model_used)
    if(args.model_used=='lstm'):
        input_size = 15000  
        hidden_size = 128  # Adjust as needed
        num_layers = 2  # Adjust as needed
        net = LSTMModel(input_size, hidden_size, num_layers, args.num_classes)
        net = net.to(device)
    elif(args.model_used=='resnet18'):
        net = resnet18(input_channels=nleads, num_classes=args.num_classes).to(device)
    elif(args.model_used=='resnet34'):
        net = resnet34(input_channels=nleads, num_classes=args.num_classes).to(device)
    elif(args.model_used=='GRU'):
        input_size = 15000
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
        enc_in = 12 
        c_out = args.num_classes           
        net = Mamba(d_model, expand, enc_in, c_out, d_conv=4, d_ff=32, e_layers=2, dropout=0.05)
        net = net.to(device)
    elif(args.model_used=='Residual_Conv_GRU' or args.model_used=='mini_Residual_Conv_GRU' or args.model_used=='Residual_Conv_GRU_test'):
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
                                num_classes=args.num_classes, num_layers=2)
        else:
            net = mini_Residual_ConvTransformer( input_size=input_size, batch_size=args.batch_size, input_channels=nleads, 
                                num_classes=args.num_classes, num_layers=2)
        net = net.to(device)        
    elif(args.model_used=='Residual_conv_retnet'):        
        input_size=15000
        net = Residual_conv_retnet( input_size=input_size, batch_size=args.batch_size, input_channels=nleads, 
                                num_classes=args.num_classes, num_layers=2,hidden_dim=64,ffn_size=128,quantize=args.quantize)
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
    dataset_name='ptb-xl'
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
        if(phase=='train'):
            model_name=f'{model_used}_{ct}_{leads}_{seed}'
        else:
            model_name=''
        model_path =model_dir+model_name
    elif(dataset_name=='ptb-xl'):
        num_classes = 5  # Set this to your number of classes
        if(phase=='train'):
            model_name=f'{model_used}_{ct}_{leads}_{seed}'
        else:
            model_name=''
        model_path =model_dir+model_name
    elif(dataset_name=='shaoxing-ninbo'):
        num_classes = 63  # Set this to your number of classes
        if(phase=='train'):
            model_name=f'{model_used}_{ct}_{leads}_{seed}'
        else:
            model_name=''
        model_path =model_dir+model_name
    
    quantize = False
    prune = False
    quantization_precision = 16  
    if quantize :
        model_path=model_dir+"ResU_Dense_2024-07-28-12:18_all_42" 
    
    if prune :
        model_path = model_dir+'Residual_Conv_GRU_2024-08-15-21:49_all_42'

    if resume:
        resume_model_path=model_dir+"Mamba_2024-07-06-13:55_all_42"   

    gpu_number=2    

args = Args()

if __name__ == "__main__":
    """
    Main script for initializing and running ECG model training and evaluation with support for different datasets,
    pruning, quantization, and various model architectures.

    Args:
        -config_file (str, optional): Path to the configuration file containing model parameters. Default is None.

    Workflow:
        1. Argument Parsing:
           The script parses arguments from the command line, allowing the user to specify a configuration file. If a config file is provided, it loads parameters such as learning rate, model path, and other settings.
        
        2. Model and Directory Setup:
           It sets up the model path and ensures the necessary directories (e.g., for models) exist. It checks if GPU is available and sets the device accordingly (CPU or GPU).
        
        3. Data Loading:
           The script loads ECG datasets (CPSC-2018, PTB-XL, Shaoxing-Ninbo) based on the selected dataset. For each dataset, the appropriate `DataLoader` is created for training, validation, and test sets.
        
        4. Model Initialization:
           The model is initialized based on the specified architecture (e.g., LSTM, ResNet, etc.). It supports model pruning and quantization if specified in the configuration.
        
        5. Training Loop:
           The model is trained using the specified optimizer (e.g., Adam) and learning rate scheduler (e.g., StepLR). Training metrics such as accuracy, precision, recall, F1 score, and balanced accuracy are calculated.
        
        6. Evaluation:
           After training, the model is evaluated on the validation dataset for each fold. Evaluation metrics are calculated similarly to the training metrics.
        
        7. Model Pruning/Quantization:
           If model pruning or quantization is enabled, the corresponding model transformation (pruning or half-precision conversion) is applied.

        8. Results Summary:
           After training and evaluation, the results (accuracy, precision, recall, F1 score) are summarized for both the training and validation phases. The summary is saved to a CSV file.

    Datasets:
        - CPSC-2018 ECG Dataset
        - PTB-XL ECG Dataset
        - Shaoxing-Ninbo ECG Dataset

    Model Architectures:
        - Supports several model architectures like LSTM, ResNet, GRU, RetNet, Mamba, Residual Convolution-based models, etc.

    Pruning/Quantization:
        - Supports model pruning and 16-bit quantization for reducing model size and improving inference efficiency.

    Attributes:
        args (Namespace): Parsed arguments or configuration settings.
        data_dir (str): Path to the directory where ECG data is stored.
        database (str): The name of the database being used.
        model_dir (str): Directory where models will be saved.
        model_path (str): Path to save or load model weights.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.

    Expected Output:
        The script prints and saves the following metrics for each fold:
        - Training and validation accuracy, precision, recall, F1 score, and balanced accuracy.
        - Model size in bits and MB.
        - A summary of the results is saved in CSV format for each fold and the average across all folds.
    """
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
                elif(tmp[0]=='model_path'):
                    val=tmp[1]+':'+tmp[2]
                else:
                    val=int(tmp[1]) if tmp[1].isnumeric()  else tmp[1]
                setattr(args,tmp[0],val)
        if(not args.prune and not args.quantize):
            args.model_name = f'{args.model_used}_{args.ct}_{args.leads}_{args.seed}.pth'
            args.model_path = args.model_dir + args.model_name    

    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

    # Set the model path if it's not already set
    if not args.model_path:
        args.model_path = f'/media/user/nvme0/ECG-modeling/ptb-xl/models/lstm_{database}_{args.leads}_{args.seed}.pth'
    
    print(args.model_path)
    # Ensure the 'models' directory exists
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    

    if(args.quantize and args.quantization_precision!=16):
        gpu_available = False
        device = torch.device('cpu')
    else:
        gpu_available=torch.cuda.is_available()
        print("use_gpu : ",gpu_available)
        device = torch.device(f'cuda:{args.gpu_number}') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    
    leads = args.leads.split(',') if args.leads != 'all' else 'all'
    nleads = len(leads) if args.leads != 'all' else 12    
    label_csv = os.path.join(data_dir, 'labels.csv')
    folds = split_data(seed=args.seed)   
    test_folds = folds[9:]    
    train_val_folds = folds[:9]    
    acc_list,precision_list,recall_list,f1_list=[],[],[],[]
    acc_list_val,precision_list_val,recall_list_val,f1_list_val=[],[],[],[]
    bal_acc_list, bal_acc_list_val=[],[]
    
    for fold_idx in range(len(train_val_folds)):
        args.best_metric = 0
        # initialize model
        net = model_initialize(nleads, args,device)
        
        if(args.prune):
            print(args.model_path)
            path = args.model_path+f'_fold{fold_idx}.pth'
            print(path)
            pruned_model = prune(args, path, device, net)
            net = pruned_model.to(device)
            net.eval()
        elif(args.quantize):
            print(args.model_path)
            path = args.model_path+f'_fold{fold_idx}.pth'
            print(path)
            net.load_state_dict(torch.load(path, map_location=device))
            net=net.to('cpu')            
            net.eval()
            quantized_model=net.half()
            net = quantized_model.to(device)
            net.eval()        
        size_model=0
        
        for param in net.parameters():
            size_model += param.numel() * param.element_size()
        print(f"model size: {size_model} / bit | {size_model / 1e6:.2f} / MB")
        model_size = size_model/ 1e6/8 #MB
        print(model_size)

        print("fold:",fold_idx+1)
        val_folds = [train_val_folds[fold_idx]]        
        train_folds = [fold for i, fold in enumerate(folds) if i != fold_idx]
        
        if(args.dataset_name == 'cpsc_2018'):
            print("CPSC-2018")
            train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        elif(args.dataset_name == 'ptb-xl'):
            print("PTB-xl dataset")
            train_dataset = PTBDataset('train', data_dir, label_csv, train_folds, leads)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_dataset = PTBDataset('val', data_dir, label_csv, val_folds, leads)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_dataset = PTBDataset('test', data_dir, label_csv, test_folds, leads)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        elif(args.dataset_name == 'shaoxing-ninbo'):
            print("shaoxing_nimbo dataset")
            train_dataset = shaoxingDataset('train', data_dir, label_csv, train_folds, leads)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_dataset = shaoxingDataset('val', data_dir, label_csv, val_folds, leads)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_dataset = shaoxingDataset('test', data_dir, label_csv, test_folds, leads)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            print("Unsupported dataset")

        if(args.model_used=='Residual_ConvTransformer'):
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
        criterion = nn.BCEWithLogitsLoss()  # or another appropriate loss function
        
        if args.phase == 'train':
            if args.resume:
                path = args.resume_model_path+f'_fold{fold_idx}.pth'
                net.load_state_dict(torch.load(path, map_location=device))
                size_model=0               
                for param in net.parameters():
                    size_model += param.numel() * param.element_size()
            for epoch in range(args.epochs):                
                (avg_accuracy, avg_precision, avg_recall, avg_f1,balanced_accuracy)=train(train_loader, net, args, criterion,
                                                                        epoch, scheduler, optimizer, device, train_dataset)
                if(epoch==args.epochs-1):
                    acc_list.append(avg_accuracy)
                    precision_list.append(avg_precision)
                    recall_list.append(avg_recall)
                    f1_list.append(avg_f1) 
                    bal_acc_list.append(balanced_accuracy)

                if (args.quantize):
                    model_path=args.model_dir+f"{args.model_name}_quantize{args.quantization_precision}_fold{fold_idx}.pth"
                elif(args.prune):
                    model_path=args.model_dir+f"{args.model_name}_prune{args.quantization_precision}_fold{fold_idx}.pth"
                else:
                    model_path=args.model_path+'_fold'+str(fold_idx)+'.pth'                
                (avg_accuracy, avg_precision, avg_recall, avg_f1, avg_inference_time,balanced_accuracy)=evaluate(val_loader, 
                                                                        net, args, criterion, device,model_path,val_dataset)                
            summary_file_name=f'{args.model_dir[:-7]}/summary{args.gpu_number}.csv'
            write_summary_csv(summary_file_name, args, avg_accuracy, 
                              avg_precision, avg_recall, avg_f1, model_path,fold_idx,
                              gpu_available, avg_inference_time, model_size, balanced_accuracy)    
            acc_list_val.append(avg_accuracy)
            precision_list_val.append(avg_precision)
            recall_list_val.append(avg_recall)
            f1_list_val.append(avg_f1) 
            bal_acc_list_val.append(balanced_accuracy)

    train_avg_acc = sum(acc_list)/len(acc_list)
    val_avg_acc = sum(acc_list_val)/len(acc_list_val)

    train_avg_precision = sum(precision_list)/len(precision_list)
    val_avg_precision = sum(precision_list_val)/len(precision_list_val)

    train_avg_recall = sum(recall_list)/len(recall_list)
    val_avg_recall = sum(recall_list_val)/len(recall_list_val)

    train_avg_f1 = sum(f1_list)/len(f1_list)
    val_avg_f1 = sum(f1_list_val)/len(f1_list_val)
    
    train_avg_bal = sum(bal_acc_list)/len(bal_acc_list)
    val_avg_bal = sum(bal_acc_list_val)/len(bal_acc_list_val)

    summary_file_name=f'{args.model_dir[:-7]}/summary{args.gpu_number}.csv'
    write_summary_csv(summary_file_name, args, train_avg_acc, 
                        train_avg_precision, train_avg_recall, train_avg_f1, "avg_of_K_folds", "avg_train",
                        gpu_available, 0, model_size,train_avg_bal)    
    
    write_summary_csv(summary_file_name, args, val_avg_acc, 
                        val_avg_precision, val_avg_recall, val_avg_f1, "avg_of_K_folds", "avg_test",
                        gpu_available, 0, model_size,val_avg_bal)  
    
    print('train')
    print('accuracy:',sum(acc_list)/len(acc_list))
    print('precision:',sum(precision_list)/len(precision_list))
    print('recall:',sum(recall_list)/len(recall_list))
    print('f1:',sum(f1_list)/len(f1_list))

    print('val')
    print('accuracy:',sum(acc_list_val)/len(acc_list_val))
    print('precision:',sum(precision_list_val)/len(precision_list_val))
    print('recall:',sum(recall_list_val)/len(recall_list_val))
    print('f1:',sum(f1_list_val)/len(f1_list_val))
                 