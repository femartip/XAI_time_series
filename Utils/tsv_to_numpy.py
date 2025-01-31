import os
import pandas as pd
import numpy as np
from load_data import normalize_data

def main(datasets: list, data_dir: str):
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        
        if not os.path.isdir(dataset_dir):
            continue 

        for file in os.listdir(dataset_dir):
            if file.endswith('.tsv'):
                file_name = file.split('.')[0]
                print(f"Generating both .npy and _normalized.npy files for {dataset}/{file_name}")
                tsv_file = os.path.join(dataset_dir, file)
                df_file_data = pd.read_csv(tsv_file, sep='\t')
                df_file_data.iloc[:,0] = df_file_data.iloc[:,0].map({sorted(df_file_data.iloc[:,0].unique())[0]:1, sorted(df_file_data.iloc[:,0].unique())[1]: 2})
                print(f"Dataset classes in the range of {sorted(df_file_data.iloc[:,0].unique())}.")
                np_file_data = df_file_data.to_numpy()
                np.save(os.path.join(dataset_dir, file_name + ".npy"), np_file_data.astype(np.float32))
                normalize_data(dataset, file_name.split('_')[1].upper())
                print("Done!")

    print("Finished.")

                
if __name__ == '__main__':
    data_dir = './data'
    datasets = os.listdir(data_dir)
    main(datasets, data_dir)

    
