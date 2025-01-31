import pandas as pd 
import os
import shutil


def main():
    ucr_df = pd.read_csv('./data/DataSummary.csv', header=0)
    ucr_df_not_bin = ucr_df.loc[ucr_df['Class'] != 2]
    ucr_not_bin_names = ucr_df_not_bin['Name'].values
    
    list_dirs = os.listdir('./data/')

    for dataset in ucr_not_bin_names:
        if dataset in list_dirs:
            dir_path = os.path.join('./data/', dataset)
            shutil.rmtree(dir_path)
            print(f"Dataset {dataset} removed as it is not a binary dataset")

    print("All non-binary datasets removed")



if __name__ == '__main__':
    """
    To select all binary UCR datasets, I have imported all.
    Then I use this script to see which ones are not bianry and remove them.
    """
    main()
