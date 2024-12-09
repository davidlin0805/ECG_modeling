import os
from glob import glob
import shutil
import wfdb
import numpy as np
import pandas as pd
from scipy.io import loadmat
import ast

class datapreprocess_cpsc_2018:
    def combine_dataset(self,data_dir, output_dir):
        """
        Combine dataset files from multiple subdirectories into a single directory.
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Iterate through each subdirectory ('g1', 'g2', ..., 'gn')
        for sub_dir in glob(os.path.join(data_dir, 'g*')):
            # Find all '.hea' and '.mat' files in subdirectories
            hea_files = glob(os.path.join(sub_dir, '*.hea'))
            mat_files = glob(os.path.join(sub_dir, '*.mat'))

            # Copy each file to the output directory
            for file_path in hea_files + mat_files:
                # Get the base name of the file
                base_name = os.path.basename(file_path)
                # Construct the destination path
                dest_path = os.path.join(output_dir, base_name)
                # Copy the file
                shutil.copy(file_path, dest_path)
        
        print(f"Dataset files copied to {output_dir}")

    def gen_reference_csv(self, data_dir, reference_csv):
        """
        Generate a reference CSV file with metadata from the ECG files.
        """
        if not os.path.exists(reference_csv):
            recordpaths = glob(os.path.join(data_dir, '*.hea'))
            results = []
            for recordpath in recordpaths:
                patient_id = os.path.basename(recordpath)[:-4]
                _, meta_data = wfdb.rdsamp(recordpath[:-4])
                sample_rate = meta_data['fs']
                signal_len = meta_data['sig_len']
                age = meta_data['comments'][0]
                sex = meta_data['comments'][1]
                dx = meta_data['comments'][2]
                age = age[5:] if age.startswith('Age: ') else np.NaN
                sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
                dx = dx[4:] if dx.startswith('Dx: ') else ''
                results.append([patient_id, sample_rate, signal_len, age, sex, dx])
            df = pd.DataFrame(data=results, columns=['patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx'])
            df.sort_values('patient_id').to_csv(reference_csv, index=None)
        print(f"Reference CSV generated at {reference_csv}")

    def gen_label_csv(self,label_csv, reference_csv, dx_dict, classes):
        """
        Generate a label CSV file with the corresponding diagnosis codes.
        """
        if not os.path.exists(label_csv):
            results = []
            df_reference = pd.read_csv(reference_csv)
            for _, row in df_reference.iterrows():
                patient_id = row['patient_id']
                dxs = [dx_dict.get(code, '') for code in row['dx'].split(',')]
                labels = [0] * len(classes)
                for idx, label in enumerate(classes):
                    if label in dxs:
                        labels[idx] = 1
                results.append([patient_id] + labels)
            df = pd.DataFrame(data=results, columns=['patient_id'] + classes)
            n = len(df)
            folds = np.zeros(n, dtype=np.int8)
            for i in range(10):
                start = int(n * i / 10)
                end = int(n * (i + 1) / 10)
                folds[start:end] = i + 1
            df['fold'] = np.random.permutation(folds)
            columns = df.columns
            df['keep'] = df[classes].sum(axis=1)
            df = df[df['keep'] > 0]
            df[columns].to_csv(label_csv, index=None)
        print(f"Label CSV generated at {label_csv}")
    def preprocess(self):
        # Set the data directory (Make sure to adjust this path to your dataset location)
        data_dir = '/media/user/nvme0/ECG-modeling/cpsc_2018/datas'

        # Set the output directory (Kaggle's writable directory)
        output_dir = data_dir

        # Define the diagnosis codes and the corresponding classes
        dx_dict = {
            '426783006': 'SNR',  # Normal sinus rhythm
            '164889003': 'AF',   # Atrial fibrillation
            '270492004': 'IAVB', # First-degree atrioventricular block
            '164909002': 'LBBB', # Left bundle branch block
            '713427006': 'RBBB', # Complete right bundle branch block
            '59118001': 'RBBB',  # Right bundle branch block
            '284470004': 'PAC',  # Premature atrial contraction
            '63593006': 'PAC',   # Supraventricular premature beats
            '164884008': 'PVC',  # Ventricular ectopics
            '429622005': 'STD',  # ST-segment depression
            '164931005': 'STE',  # ST-segment elevation
        }
        classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

        # Combine dataset files
        #self.combine_dataset('/media/user/nvme0/ECG-modeling/cpsc_2018/', output_dir)

        # Generate reference CSV
        reference_csv = os.path.join(output_dir, 'reference.csv')
        self.gen_reference_csv(output_dir, reference_csv)

        # Generate labels CSV
        label_csv = os.path.join(output_dir, 'labels.csv')
        self.gen_label_csv(label_csv, reference_csv, dx_dict, classes)

class datapreprocess_ptb:
    def combine_dataset(self, data_dir, output_dir):
        """
        Combine dataset files from multiple subdirectories into a single directory.
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate through each subdirectory in   records500
        for sub_dir in glob(os.path.join(data_dir, 'records500')):       
            # Find all '.hea' and '.dat' files in subdirectories
            sub_sub_dir = os.listdir(sub_dir)
        
            for name in sub_sub_dir:
                if(name != 'index.html'):           
                    
                    hea_files = glob(os.path.join(sub_dir, name,'*.hea'))
                    dat_files = glob(os.path.join(sub_dir, name,'*.dat'))
                    
                    # Copy each file to the output directory
                    for file_path in hea_files + dat_files:
                        # Get the base name of the file
                        base_name = os.path.basename(file_path)
                        # Construct the destination path
                        dest_path = os.path.join(output_dir, base_name)
                        # Copy the file
                        shutil.copy(file_path, dest_path)
                
        
        print(f"Dataset files copied to {output_dir}")

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv('/media/user/nvme0/ECG-modeling/ptb-xl/physionet.org/files/ptb-xl/1.0.3/'+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(self,y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    def combine_database_and_diagnosis(self, path):
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        Y['diagnostic_superclass'] = Y.scp_codes.apply(self.aggregate_diagnostic)   
        return Y
    def gen_reference_csv(self,data_dir, reference_csv, df):
        """
        Generate a reference CSV file from database csv and diagnosis.
        """
        path = '/media/user/nvme0/ECG-modeling/ptb-xl/physionet.org/files/ptb-xl/1.0.3/'        
        if not os.path.exists(reference_csv):
            results=[]
            
            for _, row in df.iterrows():    
                #print(row)
                #only use sample rate 500
                patient_id = row['patient_id']                
                recordpath = os.path.join(data_dir,row['filename_hr'][-8:])
                _, meta_data = wfdb.rdsamp(recordpath)
                sample_rate = 500
                signal_len = meta_data['sig_len']       
                sex = row['sex']
                age = row['age']
                dx = row['diagnostic_superclass']
                
                files_number=row['filename_hr'][-8:]
                results.append([patient_id, sample_rate, signal_len, age, sex, dx,files_number])
                
            

            df = pd.DataFrame(data=results, columns=['patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx','files_number'])
            df.sort_values('patient_id').to_csv(reference_csv, index=None)     
            print(df.sort_values('patient_id').head(10))
            
            
        print(f"Reference CSV generated at {reference_csv}")

    def gen_label_csv(self,label_csv, reference_csv, dx_dict, classes):
        """
        Generate a label CSV file with the corresponding diagnosis codes.
        """
        
        if not os.path.exists(label_csv):
            results = []
            df_reference = pd.read_csv(reference_csv)
            for _, row in df_reference.iterrows():
                patient_id = str(row['patient_id'])
                files_name= str(row['files_number'])

                patient_id+=files_name
                dxs=row['dx']            

                labels = [0] * len(classes)
                for idx, label in enumerate(classes):               
                    if label in dxs:
                        labels[idx] = 1
                
                results.append([patient_id]  + labels)
                
            df = pd.DataFrame(data=results, columns=['patient_id'] + classes)
            n = len(df)
            folds = np.zeros(n, dtype=np.int8)
            for i in range(10):
                start = int(n * i / 10)
                end = int(n * (i + 1) / 10)
                folds[start:end] = i + 1
            df['fold'] = np.random.permutation(folds)
            columns = df.columns
            df['keep'] = df[classes].sum(axis=1) #num of classes
            df = df[df['keep'] > 0]
            df[columns].to_csv(label_csv, index=None)
            print(df.head(10))
        print(f"Label CSV generated at {label_csv}")
    def preprocess(self):
        # Set the data directory (Make sure to adjust this path to your dataset location)
        data_dir = '/media/user/nvme0/ECG-modeling/ptb-xl/datas'
        output_dir = data_dir

        # Define the diagnosis codes and the corresponding classes
        dx_dict = {
            "'NORM'" : 'NORM', #Normal ECG
            "'MI'" : 'MI', #Myocardial Infarction
            "'STTC'" : 'STTC', #ST/T Change
            "'CD'" : 'CD', #Conduction Disturbance
            "'HYP'" : 'HYP' #Hypertrophy
        }
        #classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        classes = ['NORM','MI','STTC','CD','HYP']
        path = '/media/user/nvme0/ECG-modeling/ptb-xl/physionet.org/files/ptb-xl/1.0.3/'
        
        #combine database.csv and diagnosis        

        # Combine dataset files
        #self.combine_dataset(path, output_dir)
        df=self.combine_database_and_diagnosis(path)
        # Generate reference CSV
        reference_csv = os.path.join(output_dir, 'reference.csv')
        self.gen_reference_csv(output_dir, reference_csv, df)
        # Generate labels CSV
        label_csv = os.path.join(output_dir, 'labels.csv')
        self.gen_label_csv(label_csv, reference_csv, dx_dict, classes)

class datapreprocess_shaoxing:
    def combine_dataset(self,data_dir, output_dir):
        """
        Combine dataset files from multiple subdirectories into a single directory.
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Iterate through each subdirectory 
        for sub_dir in glob(os.path.join(data_dir, '*')):

            
            for sub_sub_dir in glob(os.path.join(data_dir, sub_dir,'*')):
                # Find all '.hea' and '.mat' files in subdirectories
                hea_files = glob(os.path.join(sub_sub_dir, '*.hea'))
                mat_files = glob(os.path.join(sub_sub_dir, '*.mat'))
                # Copy each file to the output directory
                for file_path in hea_files + mat_files:
                    # Get the base name of the file
                    base_name = os.path.basename(file_path)
                    # Construct the destination path
                    dest_path = os.path.join(output_dir, base_name)
                    # Copy the file
                    shutil.copy(file_path, dest_path)
        
        print(f"Dataset files copied to {output_dir}")

    def gen_reference_csv(self,data_dir, reference_csv):
        """
        Generate a reference CSV file with metadata from the ECG files.
        """
        if not os.path.exists(reference_csv):
            recordpaths = glob(os.path.join(data_dir, '*.hea'))
            results = []
            for recordpath in recordpaths:
                if(recordpath[-11:-4]=='JS23074' or recordpath[-11:-4]=='JS01052'): #####cant read
                    continue
                
                
                patient_id = os.path.basename(recordpath)[:-4]
                _, meta_data = wfdb.rdsamp(recordpath[:-4])            

                sample_rate = meta_data['fs']
                signal_len = meta_data['sig_len']
                age = meta_data['comments'][0]
                sex = meta_data['comments'][1]
                dx = meta_data['comments'][2]
                age = age[5:] if age.startswith('Age: ') else np.NaN
                sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
                dx = dx[4:] if dx.startswith('Dx: ') else ''
                results.append([patient_id, sample_rate, signal_len, age, sex, dx])
            df = pd.DataFrame(data=results, columns=['patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx'])
            df.sort_values('patient_id').to_csv(reference_csv, index=None)

        print(f"Reference CSV generated at {reference_csv}")

    def gen_label_csv(self, label_csv, reference_csv, dx_dict, classes):
        """
        Generate a label CSV file with the corresponding diagnosis codes.
        """
        if not os.path.exists(label_csv):
            results = []
            df_reference = pd.read_csv(reference_csv)
            for _, row in df_reference.iterrows():
                patient_id = row['patient_id']
                dxs = [dx_dict.get(code, '') for code in row['dx'].split(',')]
                labels = [0] * len(classes)
                for idx, label in enumerate(classes):
                    if label in dxs:
                        labels[idx] = 1
                results.append([patient_id] + labels)
            df = pd.DataFrame(data=results, columns=['patient_id'] + classes)
            n = len(df)
            folds = np.zeros(n, dtype=np.int8)
            for i in range(10):
                start = int(n * i / 10)
                end = int(n * (i + 1) / 10)
                folds[start:end] = i + 1
            df['fold'] = np.random.permutation(folds)
            columns = df.columns
            df['keep'] = df[classes].sum(axis=1)
            df = df[df['keep'] > 0]
            df[columns].to_csv(label_csv, index=None)
        
        print(f"Label CSV generated at {label_csv}")

    def get_dx_dict(self,):
        path='/media/user/nvme0/ECG-modeling/shaoxing-ninbo/physionet.org/files/ecg-arrhythmia/1.0.0/ConditionNames_SNOMED-CT.csv'
        df=pd.read_csv(path)
        dx_dict={}
        classes=[]
        
        for _,row in df.iterrows():
            dx_dict[str(row['Snomed_CT'])]=row['Acronym Name']
            classes.append(row['Acronym Name'])
        return dx_dict, classes
    def preprocess(self):
        # Set the data directory (Make sure to adjust this path to your dataset location)
        data_dir = '/media/user/nvme0/ECG-modeling/shaoxing-ninbo/datas'

        # Set the output directory (Kaggle's writable directory)
        output_dir = data_dir

        # Define the diagnosis codes and the corresponding classes
        dx_dict,classes = self.get_dx_dict()
        print(len(classes))

        # Combine dataset files
        #combine_dataset('/media/user/nvme0/ECG-modeling/shaoxing-ninbo/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords', output_dir)
        
        # Generate reference CSV
        reference_csv = os.path.join(output_dir, 'reference.csv')
        self.gen_reference_csv(output_dir, reference_csv)    

        # Generate labels CSV
        label_csv = os.path.join(output_dir, 'labels.csv')
        self.gen_label_csv(label_csv, reference_csv, dx_dict, classes)


def main():
    preprocessor = datapreprocess_shaoxing()
    preprocessor.preprocess()


# Run the main function
if __name__ == "__main__":
    main()
