import pickle as pkl
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from config import BASE_DIR, get_config
from helper import N2N, Features, Labels, convert_sparse_to_tensor, rank_top_k
from load import load_model_data_from_ap, load_model_data_from_csv, load_sample_data_from_ap, load_verify_data_from_ap


class Learn:
    def __init__(self, config):
        self.config = config

        self.activation = tf.nn.leaky_relu  # leaky_relu relu selu elu

        # [node_cnt, node_feat_dim], f36, binary
        self.node_feat_binary = tf.placeholder(tf.float32, name="node_feat_binary")
        self.node_feat_continus = tf.placeholder(tf.float32, name="node_feat_continuous")
        # [node_cnt, node_cnt]
        self.n2nsum_c = tf.sparse_placeholder(tf.float64, name="n2nsum_c")
        self.n2nsum_w = tf.sparse_placeholder(tf.float64, name="n2nsum_w")

        # [node_cnt,1]
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        # sample node pairs to compute the ranking loss
        self.pair_ids_src = tf.placeholder(tf.int32, shape=[1, None], name="pair_ids_src")
        self.pair_ids_tgt = tf.placeholder(tf.int32, shape=[1, None], name="pair_ids_tgt")

        (
            self.loss,
            self.trainStep,
            self.betw_pred,
            self.node_embedding,
            self.param_list,
        ) = self.build_net()

        self.saver = tf.train.Saver(max_to_keep=None)
        CONFIG = tf.ConfigProto(
            device_count={"CPU": 12},  # limit to num_cpu_core CPU usage
            inter_op_parallelism_threads=100,
            intra_op_parallelism_threads=100,
            log_device_placement=False,
        )
        CONFIG.gpu_options.allow_growth = True
        self.session = tf.Session(config=CONFIG)
        # Tensorboard
        if not os.path.exists(f"{BASE_DIR}/{self.config.model_name}/logs"):
            os.makedirs(f"{BASE_DIR}/{self.config.model_name}/logs")
        self.writer = tf.summary.FileWriter(f"{BASE_DIR}/{self.config.model_name}/logs", self.session.graph)
        self.get_summary()
        self.merged = tf.summary.merge_all()
        self.session.run(tf.global_variables_initializer())

    def build_net(self):
        # [node_feat_dim, embed_dim], embed the node features
        wb_n2l = tf.Variable(
            tf.truncated_normal(
                [self.config.node_feat_dim_1, self.config.input_embedding],
                stddev=self.config.initialization_stddev,
            ),
            tf.float32,
            name="wb_n2l",
        )
        wc_n2l = tf.Variable(
            tf.truncated_normal(
                [self.config.node_feat_dim_2, self.config.input_embedding],
                stddev=self.config.initialization_stddev,
            ),
            tf.float32,
            name="wc_n2l",
        )
        # [embed_dim, embed_dim], embed aggregated neighbors
        p_node_conv = tf.Variable(
            tf.truncated_normal(
                [self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                stddev=self.config.initialization_stddev,
            ),
            tf.float32,
            name="p_node_conv",
        )
        # for regularization
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, wb_n2l)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, wc_n2l)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, p_node_conv)
        if self.config.combineID == 0:  # 'graphsage'
            # [embed_dim, embed_dim], embed the input message, i.g., relu(WX)
            p_node_conv2 = tf.Variable(
                tf.truncated_normal(
                    [self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                    stddev=self.config.initialization_stddev,
                ),
                tf.float32,
                name="p_node_conv2",
            )
            # [3*embed_dim, embed_dim], embed the concat[w_neighbors, c_neighbors, message_self ]
            p_node_conv3 = tf.Variable(
                tf.truncated_normal(
                    [3 * self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                    stddev=self.config.initialization_stddev,
                ),
                tf.float32,
                name="p_node_conv3",
            )
            # for regularization
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, p_node_conv2)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, p_node_conv3)
        elif self.config.combineID == 1:  # GRU
            w_r = tf.Variable(
                tf.truncated_normal(
                    [2 * self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                    stddev=self.config.initialization_stddev,
                ),
                tf.float32,
                name="w_r",
            )
            u_r = tf.Variable(
                tf.truncated_normal(
                    [self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                    stddev=self.config.initialization_stddev,
                ),
                tf.float32,
                name="u_r",
            )
            w_z = tf.Variable(
                tf.truncated_normal(
                    [2 * self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                    stddev=self.config.initialization_stddev,
                ),
                tf.float32,
                name="w_z",
            )
            u_z = tf.Variable(
                tf.truncated_normal(
                    [self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                    stddev=self.config.initialization_stddev,
                ),
                tf.float32,
                name="u_z",
            )
            w = tf.Variable(
                tf.truncated_normal(
                    [2 * self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                    stddev=self.config.initialization_stddev,
                ),
                tf.float32,
                name="w",
            )
            u = tf.Variable(
                tf.truncated_normal(
                    [self.config.EMBEDDING_SIZE, self.config.EMBEDDING_SIZE],
                    stddev=self.config.initialization_stddev,
                ),
                tf.float32,
                name="u",
            )
            # for regularization
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_r)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, u_r)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_z)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, u_z)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, u)

        # [embed_dim, reg_hidden], decoder
        h1_weight = tf.Variable(
            tf.truncated_normal(
                [self.config.EMBEDDING_SIZE, self.config.REG_HIDDEN],
                stddev=self.config.initialization_stddev,
            ),
            tf.float32,
            name="h1_weight",
        )
        # [reg_hidden, 1], output weight
        last_w = tf.Variable(
            tf.truncated_normal([self.config.REG_HIDDEN, 1], stddev=self.config.initialization_stddev),
            tf.float32,
            name="last_w",
        )

        # for regularization
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, h1_weight)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, last_w)

        # [node_cnt, node_feat_dim]
        # node_size = tf.shape(self.n2nsum_c)[0]
        # [node_cnt, embed_dim], f(WX)
        input_message_1 = tf.matmul(tf.cast(self.node_feat_binary, tf.float32), wb_n2l)
        input_message_2 = tf.matmul(tf.cast(self.node_feat_continus, tf.float32), wc_n2l)
        input_message = tf.concat([input_message_1, input_message_2], axis=1)
        # [node_cnt, embed_dim], no sparse
        cur_message_layer = self.activation(input_message)
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
        cur_message_layer_JK = cur_message_layer
        lv = 0
        # max_bp_iter steps of neighbor propagation
        while lv < self.config.max_bp_iter:
            lv = lv + 1
            # [node_cnt, node_cnt]*[node_cnt, embed_dim] = [node_cnt, embed_dim]
            n2npool_c = tf.sparse_tensor_dense_matmul(
                tf.cast(self.n2nsum_c, tf.float64),
                tf.cast(cur_message_layer, tf.float64),
            )
            n2npool_w = tf.sparse_tensor_dense_matmul(
                tf.cast(self.n2nsum_w, tf.float64),
                tf.cast(cur_message_layer, tf.float64),
            )

            n2npool_c = tf.cast(n2npool_c, tf.float32)
            n2npool_w = tf.cast(n2npool_w, tf.float32)

            # [node_cnt, embed_dim] * [embedding, embedding] = [node_cnt, embed_dim], dense
            node_linear_c = tf.matmul(n2npool_c, p_node_conv)
            node_linear_w = tf.matmul(n2npool_w, p_node_conv)
            # [node_cnt, 2*embed_dim]
            node_linear = tf.concat([node_linear_w, node_linear_c], axis=1)

            if self.config.combineID == 0:  # 'graphsage'
                # [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)  # type: ignore
                # [[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                # [node_cnt, 3*embed_dim]*[3*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = self.activation(tf.matmul(merged_linear, p_node_conv3))  # type: ignore

                if self.config.JK == 1:
                    cur_message_layer_JK = tf.maximum(cur_message_layer_JK, cur_message_layer)
                elif self.config.JK == 2:
                    cur_message_layer_JK = tf.minimum(cur_message_layer_JK, cur_message_layer)
                elif self.config.JK == 3:
                    cur_message_layer_JK = tf.add(cur_message_layer_JK, cur_message_layer)

            elif self.config.combineID == 1:  # gru
                r_t = tf.nn.relu(tf.add(tf.matmul(node_linear, w_r), tf.matmul(cur_message_layer, u_r)))  # type: ignore
                z_t = tf.nn.relu(tf.add(tf.matmul(node_linear, w_z), tf.matmul(cur_message_layer, u_z)))  # type: ignore
                h_t = tf.nn.tanh(tf.add(tf.matmul(node_linear, w), tf.matmul(r_t * cur_message_layer, u)))  # type: ignore
                cur_message_layer = (1 - z_t) * cur_message_layer + z_t * h_t
                cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

                if self.config.JK == 1:
                    cur_message_layer_JK = tf.maximum(cur_message_layer_JK, cur_message_layer)
                elif self.config.JK == 2:
                    cur_message_layer_JK = tf.minimum(cur_message_layer_JK, cur_message_layer)
                elif self.config.JK == 3:
                    cur_message_layer_JK = tf.add(cur_message_layer_JK, cur_message_layer)

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        if self.config.JK == 1 or self.config.JK == 2:
            cur_message_layer = cur_message_layer_JK
        elif self.config.JK == 3:
            cur_message_layer = cur_message_layer_JK / (self.config.max_bp_iter + 1)

        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        # node embedding, [node_cnt, embed_dim]
        embed_s_a = cur_message_layer 
        # decoder, two-layer MLP
        hidden = tf.matmul(embed_s_a, h1_weight)
        last_output = self.activation(hidden)
        self.betw_pred = tf.matmul(last_output, last_w)

        # [pair_size, 1]
        labels = tf.nn.embedding_lookup(self.label, self.pair_ids_src) - tf.nn.embedding_lookup(
            self.label, self.pair_ids_tgt
        )
        preds = tf.nn.embedding_lookup(self.betw_pred, self.pair_ids_src) - tf.nn.embedding_lookup(
            self.betw_pred, self.pair_ids_tgt
        )

        # l2 regularization to reduce overfitting
        regularizer = tf.contrib.layers.l2_regularizer(self.config.scale)
        reg_term = tf.contrib.layers.apply_regularization(regularizer)
        loss = self.pairwise_ranking_loss(preds, labels) + reg_term
        trainStep = tf.train.AdamOptimizer(self.config.LEARNING_RATE).minimize(loss)

        return loss, trainStep, self.betw_pred, embed_s_a, tf.trainable_variables()

    def pairwise_ranking_loss(self, preds, labels):
        """Logit cross-entropy loss with masking."""
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=tf.sigmoid(labels))
        loss = tf.reduce_sum(loss, axis=1)
        return tf.reduce_mean(loss)

    def get_summary(self):
        for var in self.param_list:
            tf.summary.histogram(var.name, var)
        tf.summary.scalar("train_loss", self.loss)
        # tf.summary.scalar('test_loss', self.test_loss)

    def prepare_batch(self, N, bfeatures, cfeatures, labels, c_n2n, w_n2n):
        Nw_inx = np.random.choice(N, self.config.BATCH_SIZE, replace=False)
        bfeatures_batch, cfeatures_batch, labels_batch, c_n2n_batch, w_n2n_batch = [], [], [], [], []
        training_networks_size = self.config.training_networks_size
        for i in Nw_inx:
            bfeatures_batch.append(
                bfeatures[i * training_networks_size : (i + 1) * training_networks_size, :]
            ) 
            cfeatures_batch.append(cfeatures[i * training_networks_size : (i + 1) * training_networks_size, :])
            labels_batch.append(labels[i * training_networks_size : (i + 1) * training_networks_size, :])
            c_n2n_batch.append(c_n2n[i * training_networks_size : (i + 1) * training_networks_size, :])
            w_n2n_batch.append(w_n2n[i * training_networks_size : (i + 1) * training_networks_size, :])

        bfeatures_BATCH = np.vstack(bfeatures_batch) 
        cfeatures_BATCH = np.vstack(cfeatures_batch)
        labels_BATCH = np.vstack(labels_batch)
        c_n2nsum_sparse = sp.block_diag(c_n2n_batch)
        c_n2nsum_sparseTensor = convert_sparse_to_tensor(c_n2nsum_sparse)
        w_n2nsum_sparse = sp.block_diag(w_n2n_batch)
        w_n2nsum_sparseTensor = convert_sparse_to_tensor(w_n2nsum_sparse)

        return (
            bfeatures_BATCH,
            cfeatures_BATCH,
            labels_BATCH,
            c_n2nsum_sparseTensor,
            w_n2nsum_sparseTensor,
        )

    def train(self, Train_inputs, Test_inputs, Train_labels, Test_labels):
        trainloss_list, testloss_list, ite_list, topN_list = [], [], [], []
        bfeatures, cfeatures, labels, c_n2n, w_n2n = Train_inputs
        labels = labels * 100
        with open(
            f"{BASE_DIR}/{self.config.model_name}/Traning_process_record.txt",
            "w",
        ) as f:  
            f.write("\nTraning begin...")

            for it in range(self.config.MAX_ITERATION):
                (
                    bfeatures_BATCH,
                    cfeatures_BATCH,
                    labels_BATCH,
                    c_n2nsum_sparseTensor,
                    w_n2nsum_sparseTensor,
                ) = self.prepare_batch(self.config.size_training_set, bfeatures, cfeatures, labels, c_n2n, w_n2n)

                ids_src = Train_labels.pair_ids_src(self.config.BATCH_SIZE)
                ids_tgt = Train_labels.pair_ids_tgt(self.config.BATCH_SIZE)
                ids_src = ids_src.reshape(1, -1)
                ids_tgt = ids_tgt.reshape(1, -1)

                my_dict = {
                    self.node_feat_binary: bfeatures_BATCH,
                    self.node_feat_continus: cfeatures_BATCH,
                    self.n2nsum_c: c_n2nsum_sparseTensor,
                    self.n2nsum_w: w_n2nsum_sparseTensor,
                    self.label: labels_BATCH,
                    self.pair_ids_src: ids_src,
                    self.pair_ids_tgt: ids_tgt,
                }
                train_loss, _ = self.session.run([self.loss, self.trainStep], feed_dict=my_dict)

                # summary to tensorboard
                if it % 10 == 0:
                    result = self.session.run(self.merged, feed_dict=my_dict)
                    self.writer.add_summary(result, it)
                
                if not os.path.exists(f"{BASE_DIR}/{self.config.model_name}/models_save"):
                    os.makedirs(f"{BASE_DIR}/{self.config.model_name}/models_save")
                # record the training loss and validation score every 200 epoch
                if (it) % 200 == 0:
                    print("epoch:%d" % (it))
                    print("Training loss is %.4f" % train_loss)
                    self.test_loss, topN = self.Test(Test_inputs, Test_labels)
                    model_path = f"{BASE_DIR}/{self.config.model_name}/models_save/model_{it}" 
                    self.SaveModel(model_path)
                    f.write("\nTraning epoch: %d" % it)
                    f.write("\nTraning loss: %.4f" % train_loss)
                    f.write("\nTest loss: %.4f" % self.test_loss)
                    f.write("\nTest topN accuracy: %.4f" % topN)
                    # save train and test loss with pickle
                    trainloss_list.append(train_loss)
                    testloss_list.append(self.test_loss)
                    ite_list.append(it)
                    topN_list.append(topN)

                    # early stop condition
                    if config.early_stop_sgn:
                        if it > 1000 and topN > 0.8 and topN < np.mean(topN_list[-10:]):
                            print("Early stop training...")
                            break
                    
        
        # also record the traning process data in .csv
        data_array = np.array([ite_list, trainloss_list, testloss_list, topN_list]).T
        data = pd.DataFrame(data_array, columns=["ite", "train_loss", "test_loss", "topN"])
        data.to_csv(f"{BASE_DIR}/{self.config.model_name}/traning_process_record.csv")

    def Test_batch(self, Test_inputs, Test_labels):
        ## Test on batch size test networks
        bfeatures, cfeatures, labels, c_n2n, w_n2n, _, _ = Test_inputs
        labels = labels * 100
        (
            bfeatures_BATCH,
            cfeatures_BATCH,
            labels_BATCH,
            c_n2nsum_sparseTensor,
            w_n2nsum_sparseTensor,
        ) = self.prepare_batch(config.BATCH_SIZE, bfeatures, cfeatures, labels, c_n2n, w_n2n)
        ids_src = Test_labels.pair_ids_src(self.config.BATCH_SIZE)
        ids_tgt = Test_labels.pair_ids_tgt(self.config.BATCH_SIZE)
        ids_src = ids_src.reshape(1, -1)
        ids_tgt = ids_tgt.reshape(1, -1)

        my_dict = {
            self.node_feat_binary: bfeatures_BATCH,
            self.node_feat_continus: cfeatures_BATCH,
            self.n2nsum_c: c_n2nsum_sparseTensor,
            self.n2nsum_w: w_n2nsum_sparseTensor,
            self.label: labels_BATCH,
            self.pair_ids_src: ids_src,
            self.pair_ids_tgt: ids_tgt,
        }
        result, test_loss = self.session.run([self.betw_pred, self.loss], feed_dict=my_dict)
        y_pre = result.reshape(-1, 1)

        n_test = self.config.BATCH_SIZE
        score_1 = 0.0
        for n_ in range(n_test):
            predict = y_pre[
                n_ * self.config.test_networks_size : (n_ + 1) * self.config.test_networks_size,
                :,
            ]
            real = labels_BATCH[
                n_ * self.config.test_networks_size : (n_ + 1) * self.config.test_networks_size,
                :,
            ]  # type: ignore

            # K =  min(10, len([i for i in real[:, 0] if i>0]))\
            K = len([i for i in real[:, 0] if i > 0])
            # print(K)
            # Top N score
            score_1 += rank_top_k(self.config.test_networks_size, K, real, predict) / n_test

        print("Test loss is %.4f" % (test_loss))
        print("TopK score for test dataset:%3f" % (score_1))

        return test_loss, score_1

    def Test(self, Inputs, Test_labels):
        # Test on all 100 test networks
        bfeatures, cfeatures, labels, _, _, c_n2n_list, w_n2n_list = Inputs
        c_n2nsum_sparse = sp.block_diag(tuple(c_n2n_list))
        c_n2nsum_sparseTensor = convert_sparse_to_tensor(c_n2nsum_sparse)
        w_n2nsum_sparse = sp.block_diag(tuple(w_n2n_list))
        w_n2nsum_sparseTensor = convert_sparse_to_tensor(w_n2nsum_sparse)
        ids_src = Test_labels.pair_ids_src(100)
        ids_tgt = Test_labels.pair_ids_tgt(100)
        ids_src = ids_src.reshape(1, -1)  # type: ignore
        ids_tgt = ids_tgt.reshape(1, -1)  # type: ignore

        my_dict = {
            self.node_feat_binary: bfeatures,
            self.node_feat_continus: cfeatures,
            self.n2nsum_c: c_n2nsum_sparseTensor,
            self.n2nsum_w: w_n2nsum_sparseTensor,
            self.label: labels,
            self.pair_ids_src: ids_src,
            self.pair_ids_tgt: ids_tgt,
        }
        #[result] = self.session.run([self.betw_pred], feed_dict=my_dict)
        result, test_loss = self.session.run([self.betw_pred, self.loss], feed_dict=my_dict)
        y_pre = result.reshape(-1, 1)
        score_1 = 0.0
        for n_ in range(self.config.size_validation_set):
            predict = y_pre[
                n_ * self.config.test_networks_size : (n_ + 1) * self.config.test_networks_size,
                :,
            ]
            real = labels[
                n_ * self.config.test_networks_size : (n_ + 1) * self.config.test_networks_size,
                :,
            ]  
            K = len([i for i in real[:, 0] if i > 0])
            # print(K)
            # Top N score
            score_1 += rank_top_k(self.config.test_networks_size, K, real, predict) / self.config.size_validation_set

        print("Test loss is %.4f" % (test_loss))
        print("TopK score for test dataset:%3f" % (score_1))

        return test_loss, score_1  

    def Test_realworld(self):
        for dataset in ['cab50', 'ap50', 'ap100']:
            if dataset in ['ap50', 'ap100']:
                c, w, x, y = load_model_data_from_ap(dataset)
            else:
                c, w, x, y = load_model_data_from_csv(dataset)
            
            Test_features = Features([c], [w], [x], [y])
            bfeatures = Test_features.region_binary_features()
            cfeatures = Test_features.continus_features()

            Test_n2n = N2N([c], [w], p_c = self.config.c_neighbors_per, p_w = self.config.w_neighbors_per)
            c_n2n_list, _ = Test_n2n.generate_c_n2n()
            w_n2n_list, _ = Test_n2n.generate_w_n2n()

            c_n2nsum_sparse = sp.block_diag(tuple(c_n2n_list))
            c_n2nsum_sparseTensor= convert_sparse_to_tensor(c_n2nsum_sparse)  
            w_n2nsum_sparse = sp.block_diag(tuple(w_n2n_list))
            w_n2nsum_sparseTensor= convert_sparse_to_tensor(w_n2nsum_sparse)  

            my_dict={
                self.node_feat_binary: bfeatures, 
                self.node_feat_continus: cfeatures,
                self.n2nsum_c: c_n2nsum_sparseTensor,
                self.n2nsum_w: w_n2nsum_sparseTensor
            }

            result = self.session.run([self.betw_pred], feed_dict=my_dict)
            y_pre= result[0].reshape(-1,1)
            y_ranking=np.argsort(-y_pre,axis=0)

            # store the ranking result in .pickle
            with open(f"{BASE_DIR}/{self.config.model_name}/nodes_ranking_{dataset}.pickle", 'wb') as f:
                pkl.dump(y_ranking, f)
    
    def SaveModel(self, model_path):
        self.saver.save(self.session, model_path)
        print("model has been saved success!")

    def LoadModel(self, it):
        # ckpt = tf.train.get_checkpoint_state(model_path)
        # print(ckpt)
        new_saver = tf.train.import_meta_graph(f"{BASE_DIR}/{self.config.model_name}/models_save/model_{it}.meta")
        new_saver.restore(self.session, f"{BASE_DIR}/{self.config.model_name}/models_save/model_{it}")

        print("restore model from file successfully")


def prepare_inputs_train(config):
    #### -------------------------------------Train data inputs prepare------------------------------
    c_list, w_list, x_list, y_list = load_sample_data_from_ap()  # 加载训练样本
    train_features = Features(c_list, w_list, x_list, y_list)
    bfeatures = train_features.region_binary_features()
    cfeatures = train_features.continus_features()

    n2n = N2N(c_list, w_list, p_c=config.c_neighbors_per, p_w=config.w_neighbors_per)
    _, w_n2nsum = n2n.generate_w_n2n()
    _, c_n2nsum = n2n.generate_c_n2n()
    Train_labels = Labels(c_list)

    hubs_list_DIFF = []
    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for p in range(2, config.h_m + 1):
            with open(f"{BASE_DIR}/data/train/Hubs_CPLEX_list_p{p}_alpha{alpha:.1f}.pickle", "rb") as f:
                hubs_list = pkl.load(f)
                hubs_list_DIFF.append(hubs_list)

    labels_Hfrequency = Train_labels.generate_labels(hubs_list_DIFF, config.h_m)
    Train_inputs = (bfeatures, cfeatures, labels_Hfrequency, c_n2nsum, w_n2nsum)

    print("Preparing training inputs finish!")

    return Train_inputs, Train_labels

def prepare_inputs_test(config):
#### -----------------------------------Test data inputs prepare-------------------------------
    c_list_test, w_list_test, x_list_test, y_list_test, _ = load_verify_data_from_ap()

    Test_features = Features(c_list_test, w_list_test, x_list_test, y_list_test)
    Test_bfeatures = Test_features.region_binary_features()
    Test_cfeatures = Test_features.continus_features()

    Test_n2n = N2N(c_list_test, w_list_test, p_c=config.c_neighbors_per, p_w=config.w_neighbors_per)
    c_n2n_list, test_c_n2nsum = Test_n2n.generate_c_n2n()
    w_n2n_list, test_w_n2nsum = Test_n2n.generate_w_n2n()
    Test_labels = Labels(c_list_test)

    Test_hubs_list_DIFF = []
    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for p in range(2, config.h_m + 1):
            with open(f"{BASE_DIR}/data/test/Hubs_CPLEX_list_p{p}_alpha{alpha:.1f}.pickle", "rb") as f:
                hubs_list = pkl.load(f)
                Test_hubs_list_DIFF.append(hubs_list)

    Test_labels_Hfrequency = Test_labels.generate_labels(Test_hubs_list_DIFF, config.h_m)
    Test_inputs = (
        Test_bfeatures,
        Test_cfeatures,
        Test_labels_Hfrequency,
        test_c_n2nsum,
        test_w_n2nsum,
        c_n2n_list,
        w_n2n_list,
    )

    print("Preparing test inputs finish!")
    
    return Test_inputs, Test_labels


if __name__ == "__main__":
    config = get_config()
    # print(config)  

    np.random.seed(1)
    tf.set_random_seed(2)

    Train_inputs, Train_labels = prepare_inputs_train(config)
    Test_inputs, Test_labels = prepare_inputs_test(config)

    ####------------------------------------------Train the model----------------------------------------
    tf.reset_default_graph()
    model = Learn(config)
    model.train(Train_inputs, Test_inputs, Train_labels, Test_labels)
    

