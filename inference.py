import os
import torch
import torch.nn as nn
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
import wfdb
from K_fold_cross import model_initialize
def preprocessing(ecg_data):
    """
    Preprocesses the input ECG data by trimming it to the last 15,000 time steps and reshaping it to fit 
    the required input format for the model.

    Parameters:
    -----------
    ecg_data : numpy.ndarray
        Input ECG data of shape (nsteps, 12), where `nsteps` is the number of time steps and 12 represents the 
        number of ECG leads (channels).

    Returns:
    --------
    numpy.ndarray
        A preprocessed ECG data array of shape (15000, 12). If the input sequence is shorter than 15,000 time steps, 
        it will be padded with zeros at the beginning.
    """
    nsteps, _ = ecg_data.shape

    # Trim the ECG data to the last 15,000 time steps
    ecg_data = ecg_data[-15000:, :]

    # Initialize a result array of zeros with a fixed shape of (15000, 12)
    result = np.zeros((15000, 12))

    # Copy the trimmed ECG data into the result, padding if necessary
    result[-nsteps:, :] = ecg_data

    return result


def inference(model, model_name, data_file_name, device):
    """
    Performs inference on ECG data using a specified deep learning model. The function reads the ECG data, 
    preprocesses it, runs the model for prediction, and prints the results including model name, confidence scores, 
    and inference speed.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained PyTorch model used for inference.
    model_name : str
        The version or name of the model being used.
    data_file_name : str
        The path to the ECG data file (in WFDB format) used for inference.
    device : torch.device
        The device on which the model and data are loaded (e.g., 'cpu' or 'cuda').

    Returns:
    --------
    None
        The function prints the prediction results, confidence scores, model details, and inference time.
    """
    # Load the ECG data using the WFDB library
    ecg_data, _ = wfdb.rdsamp(data_file_name)

    # Preprocess the ECG data
    ecg_data = preprocessing(ecg_data)

    # Convert the ECG data to a PyTorch tensor and move it to the specified device
    ecg_data = torch.from_numpy(ecg_data.transpose()).float().to(device)
    ecg_data = ecg_data.unsqueeze(0)  # Add a batch dimension

    # Perform inference without computing gradients
    with torch.no_grad():
        start_time = time.time()

        # Pass the input data through the model
        output = model(ecg_data)

        # Compute confidence scores using sigmoid activation
        confidence = torch.sigmoid(output)

        # Generate binary predictions (True if output > 0.5, False otherwise)
        predicted = output.data > 0.5

        end_time = time.time()

        # Calculate the inference speed (in seconds)
        inference_speed = end_time - start_time

    # Print inference results
    print("Model name :", model_name)
    print(f"Date : {str(datetime.datetime.now())[:10]}-{str(datetime.datetime.now())[11:16]}")
    print(f"Model version : {model_name}")
    print(f"Data filename : {data_file_name}")
    print(f"Result : {predicted}    Confidence : {confidence}")
    print(f"Inference speed : {inference_speed}")

warnings.filterwarnings('ignore', category=FutureWarning)

class Args:
    # python inference.py -model_name  cpsc_2018/models/resnet18_2024-08-29-21:18_all_42_fold0.pth -data_file cpsc_2018/datas/A0001 
    # python inference.py -model_name ptb-xl/models/Residual_Conv_GRU_2024-08-15-22:06_all_42_fold0.pth -data_file ptb-xl/datas/00001_hr 
    # python inference.py -model_name shaoxing-ninbo/models/Residual_Conv_GRU_2024-08-15-22:07_all_42_fold0.pth -data_file shaoxing-ninbo/datas/JS00001 
    
    dataset_name='shaoxing-ninbo'
    batch_size=32
    use_gpu = True  # Set to True if you want to use GPU and it's available
    ct = str(datetime.datetime.now())[:10]+'-'+str(datetime.datetime.now())[11:16]
    model_used='Residual_Conv_GRU'
    if(dataset_name=='cpsc_2018'):  
        num_classes = 9 
    elif(dataset_name=='ptb-xl'):
        num_classes = 5          
    elif(dataset_name=='shaoxing-ninbo'):
        num_classes = 63  

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-model_name", help="model_name",dest="model_name",default=None)
    parser.add_argument("-data_file", help="data_file_name",dest="data_file_name",default=None)
    arguments = parser.parse_args()
    args =Args()
    device = torch.device(f'cuda:0') if args.use_gpu and  torch.cuda.is_available() else torch.device('cpu')
    
    model = model_initialize(12,args, device)
    model.load_state_dict(torch.load(arguments.model_name, map_location=device))
    inference(model, arguments.model_name, arguments.data_file_name, device)




