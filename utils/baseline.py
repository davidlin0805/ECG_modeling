import os
import warnings
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import wfdb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)

data_dir = '/media/user/nvme0/ECG-modeling/cpsc_2018/datas'
output_dir = data_dir
classifier = 'LR'  # Options: 'LR', 'RF', 'LGB', 'MLP', or 'all'
seed = 42

def split_data(seed=42):
    """
    Split data into training, validation, and test sets.
    """
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)
    return folds[:8], folds[8:9], folds[9:]

def prepare_input(ecg_file: str):
    """
    Prepare input data from ECG file.
    """
    if ecg_file.endswith('.mat'):
        ecg_file = ecg_file[:-4]
    ecg_data, _ = wfdb.rdsamp(ecg_file)
    nsteps, nleads = ecg_data.shape
    ecg_data = ecg_data[-15000:, :]
    result = np.zeros((15000, nleads))  # 30 s, 500 Hz
    result[-nsteps:, :] = ecg_data
    return result.transpose()

def extract_features(ecg_data):
    """
    Extract features from ECG data.
    Placeholder function: implement your feature extraction logic here.
    """
    # Example: flatten the data as a simple feature extraction method
    return ecg_data.flatten()

def generate_features_csv(features_csv, data_dir, patient_ids):
    """
    Generate a CSV file with extracted features for each patient.
    """
    print('Generating expert features...')
    ecg_features = []
    for patient_id in tqdm(patient_ids):
        ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir, patient_id))
        ecg_features.append(extract_features(ecg_data))
    df = pd.DataFrame(ecg_features, index=patient_ids)
    df.index.name = 'patient_id'
    df.to_csv(features_csv)
    return df

def find_optimal_threshold(y_true, y_score):
    """
    Find the optimal threshold for classification.
    """
    thresholds = np.linspace(0, 1, 100)
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    return thresholds[np.argmax(f1s)]

def train_and_evaluate(classifier, X_train, y_train, X_val, y_val, X_test, y_test, classes):
    """
    Train and evaluate a classifier, returning the F1 scores and thresholds.
    """
    if classifier == 'LR':
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
    elif classifier == 'RF':
        model = RandomForestClassifier(n_estimators=300, max_depth=10)
    elif classifier == 'LGB':
        model = LGBMClassifier(n_estimators=100)
    else:
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    
    if classifier != 'MLP':
        model = OneVsRestClassifier(model)

    print(f'Start training {classifier}...')
    model.fit(X_train, y_train)
    
    y_val_scores = model.predict_proba(X_val)
    y_test_scores = model.predict_proba(X_test)
    
    f1s = []
    thresholds = []
    print('Finding optimal thresholds on validation dataset...')

    for i in range(len(classes)):
        y_val_score = y_val_scores[:, i]
        threshold = find_optimal_threshold(y_val[:, i], y_val_score)
        y_test_score = y_test_scores[:, i]
        y_test_pred = y_test_score > threshold
        f1 = f1_score(y_test[:, i], y_test_pred)
        thresholds.append(threshold)
        f1s.append(f1)
    
    print(f'{classifier} F1s:', f1s)
    print('Avg F1:', np.mean(f1s))

    return f1s, thresholds

def main():
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    features_csv = os.path.join(output_dir, 'features.csv')
    labels_csv = os.path.join(output_dir, 'labels.csv')

    df_labels = pd.read_csv(labels_csv)
    patient_ids = df_labels['patient_id'].tolist()
    
    if not os.path.exists(features_csv):
        df_X = generate_features_csv(features_csv, data_dir, patient_ids)
    else:
        df_X = pd.read_csv(features_csv)

    df_X = df_X.merge(df_labels[['patient_id', 'fold']], on='patient_id')
    
    train_folds, val_folds, test_folds = split_data(seed=seed)
    feature_cols = df_X.columns[1:-1]  # Remove patient id and fold

    X_train = df_X[df_X['fold'].isin(train_folds)][feature_cols].to_numpy()
    X_val = df_X[df_X['fold'].isin(val_folds)][feature_cols].to_numpy()
    X_test = df_X[df_X['fold'].isin(test_folds)][feature_cols].to_numpy()

    y_train = df_labels[df_labels['fold'].isin(train_folds)][classes].to_numpy()
    y_val = df_labels[df_labels['fold'].isin(val_folds)][classes].to_numpy()
    y_test = df_labels[df_labels['fold'].isin(test_folds)][classes].to_numpy()

    if classifier == 'all':
        classifiers = ['LR', 'RF', 'LGB', 'MLP']
    else:
        classifiers = [classifier]

    for clf in classifiers:
        train_and_evaluate(clf, X_train, y_train, X_val, y_val, X_test, y_test, classes)

if __name__ == "__main__":
    main()
