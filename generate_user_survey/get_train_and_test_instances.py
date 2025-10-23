import json
from typing import Tuple, Dict, List

from generate_user_survey.find_prototypes import select_prototypes
from generate_user_survey.test_selection import select_test_examples
from generate_user_survey.configurations import loyalty_value_for_each_dataset

from simplifications import get_RDP_simplification

from Utils.load_data import load_dataset

import numpy as np
import os
import random

NORM = True
dataset_extra = "" if not NORM else f"_normalized"
def make_and_save_train_test_instance(dataset):
    prototypes_label_idx = select_prototypes(dataset_name=dataset)
    for c in prototypes_label_idx.keys():
        np.save(f"generate_user_survey/prototype_and_test/{dataset}_train_label_idx_{c}.npy", prototypes_label_idx[c])
    print(dataset)
    print("TRAIN")
    print("Selected proto:",prototypes_label_idx)
    test_label_idx = select_test_examples(dataset_name=dataset)

    # Make random order!
    my_random = random.Random(42)
    all_test_examples_idx = [(idx,c) for c in test_label_idx.keys() for idx in test_label_idx[c]]
    my_random.shuffle(all_test_examples_idx)

    test_idx = [idx for (idx,c) in all_test_examples_idx]
    idx_to_class = {int(idx):int(c) for (idx,c) in all_test_examples_idx}
    np.save(f"generate_user_survey/prototype_and_test/{dataset}_test_idx.npy", test_idx)
    with open(f"generate_user_survey/prototype_and_test/{dataset}_test_idx_to_class.json", "w") as f:
        json.dump(idx_to_class,f, indent=4)

    print("TEST")
    print(test_idx)

def need_to_train_and_test(dataset):
    have_all = True
    have_all &= os.path.isfile(f"generate_user_survey/prototype_and_test/{dataset}_train_label_idx_0.npy")
    have_all &= os.path.isfile(f"generate_user_survey/prototype_and_test/{dataset}_train_label_idx_1.npy")
    have_all &= os.path.isfile(f"generate_user_survey/prototype_and_test/{dataset}_test_idx.npy")
    have_all &= os.path.isfile(f"generate_user_survey/prototype_and_test/{dataset}_test_idx_to_class.json")
    return not have_all




def get_train_and_test_index(dataset, remake=False)->Tuple[Dict[str,np.ndarray], np.ndarray]:
    if remake or need_to_train_and_test(dataset):
        make_and_save_train_test_instance(dataset)

    train_idx = {}
    for c in [0,1]:
        train_idx[c] = np.load(f"generate_user_survey/prototype_and_test/{dataset}_train_label_idx_{c}.npy")
    test_idx = np.load(f"generate_user_survey/prototype_and_test/{dataset}_test_idx.npy")
    return train_idx, test_idx

def simplify_instance_to_loyalty_level(instances: List[float], dataset:str,loyalty_level:str,verbose:bool=False) -> np.ndarray:
    alpha, num_segments = loyalty_value_for_each_dataset(dataset)[loyalty_level]

    simplified_segTS = get_RDP_simplification(np.array(instances),epsilon=alpha)
    simplified_instances = np.array([segTS.line_version for segTS in simplified_segTS])
    return simplified_instances


def get_train_and_test_instances(dataset, loyalty_level)->Tuple[Dict[str,np.ndarray], np.ndarray]:
    train_ts = load_dataset(dataset, data_type=f"TRAIN{dataset_extra}")
    test_ts = load_dataset(dataset, data_type=f"TEST{dataset_extra}")
    train_instances, test_instances = get_train_and_test_index(dataset)
    full_train_instances = {}
    for c in train_instances.keys():
        full_train_instances[c] = train_ts[train_instances[c]]

    full_test_instances =  test_ts[test_instances]
    full_train_instances_simplified= {
        0: simplify_instance_to_loyalty_level(full_train_instances[0], dataset, loyalty_level),
        1: simplify_instance_to_loyalty_level(full_train_instances[1], dataset, loyalty_level)}
    full_test_instances_simplified = simplify_instance_to_loyalty_level(full_test_instances, dataset, loyalty_level)
    return full_train_instances_simplified, full_test_instances_simplified