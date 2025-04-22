import numpy as np
import pandas as pd
import os 
import math 

def generate_time_series2(num_samples: int) -> np.ndarray:
    """
    Here is a simple way to force low AUC. Do not train a new classifier but hard-code one to take a ts on x timesteps as input,
    and hard-code it to have k classes, and then simply count the number odd(ts) of timesteps ts_i that have integer value floor(ts_i)
    being odd, and then let the classifier output class odd(ts) mod k. Then test the classifier on a dataset on x timesteps, 
    for various values of x, running the normal simplification algorithms, doing interpolation on the simplified series to calculate x values in 
    the normal way and compute AUC in the normal way. The resulting plots should be flat and lower for higher values of k. Agree?
    """
    np.random.seed(42)
    data = []
    x = 80  #Num timesteps
    k = 40 #Num classes

    for i in range(num_samples):
        time_series = np.random.uniform(0,100,size=x)
        odd_flour_count = sum([0 if math.floor(x)%2==0 else 1 for x in time_series])
        label = odd_flour_count%k
        data.append(np.insert(time_series,0,label))

    return np.array(data)


def generate_time_series(num_samples: int) -> np.ndarray:
    np.random.seed(42)
    low_range = (1,10)
    high_range = (20,30)
    data = []
    num_samples = round(num_samples/2)
    data_points = 60

    for _ in range(num_samples):
        low_values = np.random.uniform(low_range[0], low_range[1], size=5)
        high_values = np.random.uniform(high_range[0], high_range[1], size=5)
        
        class_1_series = []
        for i in range(data_points):
            if i % 2 == 0: 
                class_1_series.append(np.random.choice(high_values))
            else:
                class_1_series.append(np.random.choice(low_values))
        
        class_2_series = []
        for i in range(data_points):
            if i % 2 == 0:  
                class_2_series.append(np.random.choice(low_values))
            else:
                class_2_series.append(np.random.choice(high_values))
        
        data.append(np.array([1]+ class_1_series))
        data.append(np.array([2] + class_2_series))
    
    return np.array(data)

def normalize_data(dataset: np.ndarray) -> np.ndarray:
    labels = dataset[:,0]
    dataset = dataset[:,1:]
    max_over_all = np.max(dataset)
    min_over_all = np.min(dataset)
    
    dataset = (dataset - min_over_all) / (max_over_all - min_over_all + 1e-8)
    
    dataset = np.hstack((labels.reshape(-1, 1), dataset))
    assert not np.isnan(dataset).any(), f"NaN values in the dataset {dataset}."
    return dataset

if __name__ == '__main__':
    time_series_1 = False
    if time_series_1:
        assert os.path.isdir("data/Synthetic_1")
        #Train
        time_series_dataset = generate_time_series(num_samples=200)
        np.save("data/Synthetic/Synthetic_TRAIN.npy", time_series_dataset)
        time_series_dataset_norm = normalize_data(time_series_dataset)
        print(time_series_dataset_norm.shape)
        print(time_series_dataset_norm)
        np.save("data/Synthetic/Synthetic_TRAIN_normalized.npy", time_series_dataset_norm)

        #Validation
        validation_dataset = generate_time_series(num_samples=100)
        np.save("data/Synthetic/Synthetic_VALIDATION.npy", validation_dataset)
        validation_norm = normalize_data(validation_dataset)
        print(validation_norm.shape)
        print(validation_norm)
        np.save("data/Synthetic/Synthetic_VALIDATION_normalized.npy", validation_norm)
        
        #Test
        test_dataset = generate_time_series(num_samples=100)
        np.save("data/Synthetic/Synthetic_TEST.npy", test_dataset)
        test_norm = normalize_data(test_dataset)
        print(test_norm.shape)
        print(test_norm)
        np.save("data/Synthetic/Synthetic_TEST_normalized.npy", test_norm)
    else:
        assert os.path.isdir("data/Synthetic2")
        time_series_dataset = generate_time_series2(num_samples=100)
        print(time_series_dataset)
        #Train
        np.save("data/Synthetic2/Synthetic2_TRAIN.npy", time_series_dataset)
        time_series_dataset_norm = normalize_data(time_series_dataset)
        print(time_series_dataset_norm.shape)
        print(time_series_dataset_norm)
        np.save("data/Synthetic2/Synthetic2_TRAIN_normalized.npy", time_series_dataset_norm)

        #Validation
        validation_dataset = generate_time_series2(num_samples=100)
        np.save("data/Synthetic2/Synthetic2_VALIDATION.npy", validation_dataset)
        validation_norm = normalize_data(validation_dataset)
        print(validation_norm.shape)
        print(validation_norm)
        np.save("data/Synthetic2/Synthetic2_VALIDATION_normalized.npy", validation_norm)
        
        #Test
        test_dataset = generate_time_series2(num_samples=100)
        np.save("data/Synthetic2/Synthetic2_TEST.npy", test_dataset)
        test_norm = normalize_data(test_dataset)
        print(test_norm.shape)
        print(test_norm)
        np.save("data/Synthetic2/Synthetic2_TEST_normalized.npy", test_norm)