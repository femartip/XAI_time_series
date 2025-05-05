from simplifications import get_OS_simplification, get_RDP_simplification, get_bottom_up_simplification, \
    get_VW_simplification, get_LSF_simplification
from  pythonServer.simplification import simplify_ts

from SearchOptimalValue.binarySearchWLocal import binary_search_function_with_bruteforce
from Utils.load_data import load_dataset, load_dataset_labels
from Utils.metrics import calculate_kappa_loyalty
from Utils.load_models import model_batch_classify

DATASET_PREDS = {}
DATASET_CLASSES = {}
def eval_function(algo, datset_name, model_path, target_loyalty, alpha):
    # Search for correct loyalty
    global DATASET_PREDS, DATASET_CLASSES
    all_ts = load_dataset(dataset_name=datset_name, data_type="TEST_normalized")
    if datset_name not in DATASET_CLASSES:
        num_classes = len(set(load_dataset_labels(dataset_name=datset_name, data_type="TEST_normalized")))
        DATASET_CLASSES[datset_name] = num_classes

    num_classes = DATASET_CLASSES[datset_name]
    if datset_name not in DATASET_PREDS:
        all_preds = model_batch_classify(model_path, all_ts.tolist(), num_classes)
        DATASET_PREDS[datset_name] = all_preds

    all_preds = DATASET_PREDS[datset_name]
    all_simplifications = [simplify_ts(algo=algo, alpha=alpha, time_series=ts) for ts in all_ts]
    all_simp_preds = model_batch_classify(model_path=model_path,batch_of_timeseries=all_simplifications, num_classes=num_classes)

    kappa_loyalty = calculate_kappa_loyalty(all_preds, all_simp_preds, num_classes)
    return abs(kappa_loyalty-target_loyalty)

DATASET = {}
def eval_function_complexity(algo, datset_name, alpha):
    global DATASET
    if datset_name not in DATASET:
        DATASET[datset_name] = load_dataset(dataset_name=datset_name, data_type="TEST_normalized")
    all_ts = DATASET[datset_name]
    all_simplifications = [simplify_ts(algo=algo, alpha=alpha, time_series=ts,segSTSVersion=True) for ts in all_ts]
    complexity_of_simps = list(map(lambda simp: len(simp.x_pivots), all_simplifications))
    avg_complexity = sum(complexity_of_simps) / len(complexity_of_simps)
    return avg_complexity


def find_alpha_giving_target_complexity( algo:str, dataset_name:str, target_complexity):
    function_to_be_passed = lambda alpha: eval_function_complexity(algo=algo, datset_name=dataset_name,
                                                        alpha=alpha)
    best_overall_alpha = binary_search_function_with_bruteforce(
        f=function_to_be_passed,
        y=target_complexity,
        left=0,
        right=10**3,
        max_iterations=20,
        bruteforce_points=10
    )
    print("best_overall_alpha:", best_overall_alpha)
    print("dist to target:", abs(function_to_be_passed(best_overall_alpha)-target_complexity))


def findBestLoyalty(maxComplexity:float, model_path:str, algo:str, dataset_name:str):
    all_ts = load_dataset(dataset_name=dataset_name, data_type="TEST_normalized")
    function_to_be_passed = lambda alpha : eval_function(algo=algo, datset_name=dataset_name,model_path=model_path, alpha=alpha)
    best_overall_alpha = binary_search_function_with_bruteforce(
        f=function_to_be_passed,
        y = 1,
        left=0,
        right=10**10


    )
if __name__ == "__main__":
    import time
    dataset = "ItalyPowerDemand"
    algos = ["BU"]
    time_spent = {}
    for algo in algos:
        start = time.time()
        print(find_alpha_giving_target_complexity(algo, dataset, target_complexity=5.0001))
        end = time.time()
        time_spent[algo] = end-start

    for key, value in time_spent.items():
        print(key, value)
