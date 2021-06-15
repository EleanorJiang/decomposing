from mi.mixed import *
import numpy as np
import pandas as pd
import os
from mi.util import logger


def read_csv(dir, csv_file):
    df = pd.read_csv(os.path.join(dir,csv_file), sep='\t')
    # Step 1: Generate X labels, the input r.v.s, now 14 in total
    power_range = np.arange(1.25, 3, 0.25)
    x_labels = ["log_prob",
                "baseline",
                "rolling_average"]
    for i in power_range:
        name = 'rolling_lp_power' + str(i)
        x_labels.append(name)
    # Step 2: Extract those values from df
    df["baseline"] = df["freq"] * df["word_len"]
    return pd.DataFrame(df, columns=x_labels), df['time'].to_numpy()


def calculate_vinfo_gausian(df, all_Y, key, transform_func, k, drop_rate=0.1, seed=1234):
    # Step 3: For every x_label, calculate v-information (linear_gausian)
    x_labels = df.columns.to_list()
    v_info_dict = dict.fromkeys(x_labels, [0,0]) # scalars
    tmp_v_infos = dict.fromkeys(x_labels, [])  # lists
    tmp_v_infos_ori = dict.fromkeys(x_labels, [])  # lists
    # randomly split train/test set
    for i in range(1):
        # Randomly sampled drop_rate samples from all_X
        np.random.seed(seed+i)
        idx = np.array([i for i in range(all_Y.shape[0])])  # 0- 659
        chosen_idx = np.random.choice(idx, int(all_Y.shape[0] * (1- drop_rate)), replace=False)  # choose drop_rate out of 660
        Y = all_Y[chosen_idx]
        test_Y = all_Y[list(set(idx) - set(chosen_idx))]
        for x_label in x_labels:
            all_X = df[x_label].to_numpy()
            X = all_X[chosen_idx]
            test_X = all_X[list(set(idx) - set(chosen_idx))]
            v_info, v_info_ori = F_gaussian(X, Y, test_X, test_Y, transform_func, k=k, x_label= x_label, key=key, id=i)
            logger.info("X: {}\tModel: {}-{}\tVI: {}\tVI_ori: {}\tid: {}".format(x_label, key, k, v_info, v_info_ori, i))
            tmp_v_infos[x_label].append(v_info)
            tmp_v_infos_ori[x_label].append(v_info_ori)
            v_info_dict[x_label][0] = sum(tmp_v_infos[x_label])/len(tmp_v_infos[x_label])
            v_info_dict[x_label][1] = sum(tmp_v_infos_ori[x_label])/len(tmp_v_infos_ori[x_label])
    return v_info_dict

if __name__ == '__main__':
    dir = "/cluster/work/sachan/ct/data/decomposing"
    os.chdir('/cluster/work/sachan/ct/data/decomposing')
    csv_file = "reading_times_df.csv"
    result_file = "ns_vinfo_results.csv"
    transform_funcs = {  #"super": transform_super,
                          #  "log": transform_log,
                            "exp": transform_exp
                         # "poly": transform_poly,
                         # "log_poly": transform_log_poly
                          }
    df, Y = read_csv(dir, csv_file)  # Y is numpy array (reading time) of length N
    v_info_df = pd.DataFrame()
    for key,transform_func in transform_funcs.items():
        for k in np.arange(1, 2):
            v_info_dict = calculate_vinfo_gausian(df, Y, key, transform_func, k=k, drop_rate=0.1)
            v_info_df = pd.DataFrame.from_dict(v_info_dict)
            v_info_df["model"] = key
            v_info_df["order"] = k
            if os.path.isfile(result_file):
                df = pd.read_csv(result_file, delimiter=',')
                df = pd.concat([df, v_info_df], ignore_index=True)
            df.to_csv(result_file)
