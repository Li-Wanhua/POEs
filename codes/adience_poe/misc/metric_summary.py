import os
import argparse
import numpy as np


def process_one_exp(path):
    out_ls = []
    for fold_dir in os.listdir(path):
        npy_path = os.path.join(path, fold_dir, 'test_result.npy') 
        if os.path.exists(npy_path) and os.path.isfile(npy_path):
            test_result_mat = np.load(npy_path)
            out_ls.append(test_result_mat)
        
    if len(out_ls) > 0:
        results_mat = np.vstack(out_ls)
        print(results_mat.shape)
        means = results_mat.mean(axis=0)
        stds = results_mat.std(axis=0)
        results = [(mean, std) for mean, std in zip(means, stds)]
        print("{}".format(path))
        for k, val in zip(["mae", "acc"], results[1:-1:1]):
            print("{}: {}+-{}".format(k, *val))


#%%
if __name__ == "__main__":
    AP = argparse.ArgumentParser()
    AP.add_argument("--logdir", type=str, default="./log")
    args = AP.parse_args()     

    os.chdir(args.logdir)
    for root_dir in os.listdir('.'):
        if os.path.isdir(root_dir):
            process_one_exp(root_dir)
