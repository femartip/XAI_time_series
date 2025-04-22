from pythonServer.NativeGuide.find_native_guide import find_native_cf
from pythonServer.KerasModels.load_keras_model import no_save_batch_classify,model_classify
import numpy as np
import itertools


def findsubsets(s, n):
    all_subsets = []
    for i in range(len(s) - n + 1):
        curr_sub = []
        for j in range(n):
            curr_sub.append(i + j + 1)
        if len(curr_sub) != n:
            break
        all_subsets.append(curr_sub)
    return all_subsets


def CF_sbuseq(ts, dataset, model_name):
    org_class = model_classify(model_name,ts)
    native = find_native_cf(instance=ts, dataset_name=dataset, model_name=model_name)
    all_positions = list(range(len(native)))
    for size in range(1, len(native)):
        all_new_sizes = []
        for sub_set in findsubsets(all_positions, size):
            new_ts = []
            for i in range(len(native)):
                if i in sub_set:
                    new_ts.append(native[i])
                else:
                    new_ts.append(ts[i])

            new_ts = np.array(new_ts)
            all_new_sizes.append(new_ts)
        all_new_sizes = np.array(all_new_sizes)
        all_preds = no_save_batch_classify( model_name=model_name, batch_of_timeseries=all_new_sizes)
        if any(all_preds != org_class):
            print(len(all_preds), "preds")
            print(len(all_new_sizes), "new ts")
            idx = [c for c,pred in enumerate(all_preds) if pred != org_class][0]
            print(all_new_sizes[idx])
            return all_new_sizes[idx]

            break
        else:
            continue
        if _classify(model_name=model_name, time_series=new_ts) != org_class:
            return new_ts
    print("RETURNED FULL NATIVE")
    return native

def find_subseq_cf(ts, dataset, model_name):
    return CF_sbuseq(ts, dataset, model_name)
if __name__ == "__main__":
    print(findsubsets([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))

