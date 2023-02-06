import pickle

import numpy as np
import pandas as pd

from config import BASE_DIR


def load_sample_data_from_ap():
    """加载训练样本"""
    with open(f"{BASE_DIR}/data/train/distance_list.pickle", "rb") as f:
        c = pickle.load(f)
    with open(f"{BASE_DIR}/data/train/demand_list.pickle", "rb") as f:
        w = pickle.load(f)
    with open(f"{BASE_DIR}/data/train/x_list.pickle", "rb") as f:
        x = pickle.load(f)
    with open(f"{BASE_DIR}/data/train/y_list.pickle", "rb") as f:
        y = pickle.load(f)
    return c, w, x, y


def load_verify_data_from_ap():
    """加载测试样本"""
    with open(f"{BASE_DIR}/data/test/distance_list.pickle", "rb") as f:
        c = pickle.load(f)
    with open(f"{BASE_DIR}/data/test/demand_list.pickle", "rb") as f:
        w = pickle.load(f)
    with open(f"{BASE_DIR}/data/test/x_list.pickle", "rb") as f:
        x = pickle.load(f)
    with open(f"{BASE_DIR}/data/test/y_list.pickle", "rb") as f:
        y = pickle.load(f)
    with open(f"{BASE_DIR}/data/test/obj_CPLEX_list_p%d_alpha%.1f.pickle" % (4, 0.5), "rb") as f:
        objs = pickle.load(f)
    return c, w, x, y, objs


def load_model_data_from_csv(dataset):
    fn = pd.read_csv(f"{BASE_DIR}/data/instances/{dataset}_nodes.csv")
    fc = pd.read_csv(f"{BASE_DIR}/data/instances/{dataset}_c.csv")  # TODO 下面的是 c_dataset 没有统一
    fw = pd.read_csv(f"{BASE_DIR}/data/instances/{dataset}_w.csv")

    x, y = np.array(fn.latitude.tolist()), np.array(fn.longitude.tolist())
    name = fn.ID.tolist()
    n = len(name)
    name_inv = {}
    for i in range(n):
        name_inv[name[i]] = i

    c = np.zeros((n, n))
    w = np.zeros((n, n))
    if dataset == "cab100":
        for row in fc.itertuples():
            c[row.fromnode][row.tonode] = row.c
        for row in fw.itertuples():
            w[row.fromnode][row.tonode] = row.w
    else:
        for row in fc.itertuples():
            c[name_inv[row.fromnode]][name_inv[row.tonode]] = row.c
        for row in fw.itertuples():
            w[name_inv[row.fromnode]][name_inv[row.tonode]] = row.w

    return c, w, x, y


def load_model_data_from_ap(dataset):
    with open(f"{BASE_DIR}/data/instances/c_{dataset}.pickle", "rb") as file:
        c = pickle.load(file)
    with open(f"{BASE_DIR}/data/instances/w_{dataset}.pickle", "rb") as file:
        w = pickle.load(file)
    with open(f"{BASE_DIR}/data/instances/x_{dataset}.pickle", "rb") as file:
        x = pickle.load(file)
    with open(f"{BASE_DIR}/data/instances/y_{dataset}.pickle", "rb") as file:
        y = pickle.load(file)

    return c, w, x, y
