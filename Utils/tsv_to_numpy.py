import os
import pandas as pd
import numpy as np
from load_data import normalize_data

def main(datasets: list, data_dir: str):
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        
        if not os.path.isdir(dataset_dir):
            continue 
        
        files_list = os.listdir(dataset_dir)
        if f"{dataset}_VALIDATION.npy" not in files_list:
            print(f"Splitting {dataset} training set into 80% training and 20% validation.")
            train_file = [file for file in os.listdir(dataset_dir) if file.endswith('_TRAIN.tsv')][0]
            train_data = pd.read_csv(os.path.join(dataset_dir, train_file), sep='\t')
            train_data = train_data.sample(frac=0.8)
            validation_data = train_data.sample(frac=0.2)
            train_data = train_data.drop(validation_data.index)
            train_data.to_csv(os.path.join(dataset_dir, train_file), sep='\t', index=False)
            validation_data.to_csv(os.path.join(dataset_dir, train_file.replace('TRAIN', 'VALIDATION')), sep='\t', index=False)
        

        for file in os.listdir(dataset_dir):
            if file.endswith('.tsv'):
                file_name = file.split('.')[0]
                print(f"Generating both .npy and _normalized.npy files for {dataset}/{file_name}")
                tsv_file = os.path.join(dataset_dir, file)
                df_file_data = pd.read_csv(tsv_file, sep='\t')
                
                unique_classes = sorted(df_file_data.iloc[:, 0].unique())
                min_class = min(unique_classes)
                print(f"Dataset classes in the range of {unique_classes}.")
                if min_class != 0:
                    class_mapping = {cls: idx + 1 for idx, cls in enumerate(unique_classes)}
                    df_file_data.iloc[:, 0] = df_file_data.iloc[:, 0].map(class_mapping)
                
                print(f"Dataset classes in the range of {sorted(df_file_data.iloc[:,0].unique())}.")
                np_file_data = df_file_data.to_numpy()
                np.save(os.path.join(dataset_dir, file_name + ".npy"), np_file_data.astype(np.float32))
                try:
                    normalize_data(dataset, file_name.split('_')[1].upper())
                except Exception as e:
                    print(f"Error while normalizing {dataset}/{file_name}. Error: {e}")
                    continue
                print("Done!")

    print("Finished.")

                
if __name__ == '__main__':
    """
    Main function to convert tsv files to numpy files.
    If the dataset does not have a validation set, it will split the training set into 80% training and 20% validation.
    It also normalizes the data and saves it as _normalized.npy.
    """
    data_dir = './data'
    datasets = os.listdir(data_dir)
    main(datasets, data_dir)

    
