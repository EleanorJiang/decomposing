from mi.mixed import *
from mi.kde import mutual_kde
import numpy as np
import pandas as pd
import os
# from mi.util import logger1
from exp_vinfo import read_csv


def calculate_mi(df, all_Y, key, func, drop_rate=0.1, seed=1234):
    # Step 3: For every x_label, calculate v-information (linear_gausian)
    x_labels = df.columns.to_list()
    v_info_dict = dict.fromkeys(x_labels, 0) # scalars
    tmp_v_infos = dict.fromkeys(x_labels, [])  # lists
    # randomly split train/test set
    for i in range(1):
        # Randomly sampled drop_rate samples from all_X
        np.random.seed(seed+i)
        idx = np.array([i for i in range(all_Y.shape[0])])  # 0- 659
        chosen_idx = np.random.choice(idx, int(all_Y.shape[0] * (1- drop_rate)), replace=False)  # choose drop_rate out of 660
        Y = all_Y[chosen_idx]
        for x_label in x_labels:
            all_X = df[x_label].to_numpy()
            X = all_X[chosen_idx]
            v_info = func(X, Y)
            print("X: {}\tMI_Type: {}\tMI_Value: {}\tid: {}".format(x_label, key, v_info, i))
            tmp_v_infos[x_label].append(v_info)
            v_info_dict[x_label] = sum(tmp_v_infos[x_label])/len(tmp_v_infos[x_label])
    return v_info_dict

if __name__ == '__main__':
    os.chdir('/cluster/work/sachan/ct/data/decomposing')
    dir = "/cluster/work/sachan/ct/data/decomposing"
    csv_file = "reading_times_df.csv"
    funcs = { #"Mixed_KSG": Mixed_KSG, "Partition": Partition, "Noisy_KSG": Noisy_KSG, "KSG": KSG,
             "KDE": mutual_kde}
    df, Y = read_csv(dir, csv_file) # Y is numpy array (reading time) of length N
    v_info_df = pd.DataFrame()
    for key,func in funcs.items():
        v_info_dict = calculate_mi(df, Y, key, func, drop_rate=0.1)
        # v_info_dict["model"] = key
        # v_info_df = v_info_df.append(v_info_dict, ignore_index=True)
        # v_info_df.to_csv("ns_mi_results.csv")
