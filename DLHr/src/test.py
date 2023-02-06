import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from learn import Learn, prepare_inputs_test
from config import BASE_DIR, get_config


if __name__ == "__main__":
    config = get_config()
    print(config)  # 打印当前配置

    np.random.seed(1)
    tf.set_random_seed(2)

    Test_inputs, Test_labels = prepare_inputs_test(config)
    
    ####------------------------------------------Load model and test-------------------------------------
    # the load epoch
    it = 4000
    tf.reset_default_graph()
    model = Learn(config)
    model.LoadModel(it)
    # test on synthetic networks
    model.Test(Test_inputs, Test_labels)
    # test on realworld datasets
    model.Test_realworld()

    
