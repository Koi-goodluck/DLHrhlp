import numpy as np
import tensorflow as tf


class Features:
    def __init__(self, c_list, w_list, x_list, y_list):

        self.num_networks = len(c_list)
        self.n = np.shape(c_list[0])[0]
        self.c_list = c_list
        self.w_list = w_list
        self.x_list = x_list
        self.y_list = y_list

    def region_binary_features(self, n_features=36, a0=10):
        features_array = np.zeros((int(self.n * self.num_networks), n_features))
        count = 0
        for x_n, y_n, w_n in zip(self.x_list, self.y_list, self.w_list):
            ODi = np.sum(w_n, axis=1) + np.sum(w_n, axis=0)
            x_c = np.dot(ODi, x_n) / np.sum(ODi)
            y_c = np.dot(ODi, y_n) / np.sum(ODi)
            for x, y in zip(x_n, y_n):
                dety = y - y_c
                detx = x - x_c

                tmp = np.arctan(dety / detx) * 180 / np.pi
                if detx > 0:
                    if dety >= 0:
                        a = tmp
                    else:
                        a = abs(tmp) + 90
                elif detx < 0:
                    if dety <= 0:
                        a = tmp + 180
                    else:
                        a = abs(tmp) + 270
                else:  # detx == 0
                    if dety >= 0:
                        a = 0
                    else:
                        a = 270

                initial_angle = 0
                for f in range(n_features):
                    if initial_angle <= a < initial_angle + a0:
                        features_array[count, f] = 1

                    initial_angle += a0

                count += 1

        return features_array

    # distance from the center of gravity, Oi+Di, Ci
    def continus_features(self, n_features=3):
        features_array = np.zeros((int(self.n * self.num_networks), n_features))
        count = 0
        for x_n, y_n, c_n, w_n in zip(self.x_list, self.y_list, self.c_list, self.w_list):

            # caculate Ci
            Ci = np.sum(c_n, axis=1)
            # caculate Oi and Di
            ODi = np.sum(w_n, axis=1) + np.sum(w_n, axis=0)
            x_c = np.dot(ODi, x_n) / np.sum(ODi)
            y_c = np.dot(ODi, y_n) / np.sum(ODi)
            d_c = [np.sqrt(np.square(x_n[i] - x_c) + np.square(y_n[i] - y_c)) for i in range(self.n)]

            d_c_l = max(d_c) - min(d_c)
            ODi_l = max(ODi) - min(ODi)
            Ci_l = max(Ci) - min(Ci)
            for d_c_i, ODi_i, Ci_i in zip(d_c, ODi, Ci):
                features_array[count, 0] = (d_c_i - min(d_c)) / d_c_l
                features_array[count, 1] = (ODi_i - min(ODi)) / ODi_l
                features_array[count, 2] = (Ci_i - min(Ci)) / Ci_l

                count += 1

        return features_array


class N2N:
    EPSILON = 1e-6

    def __init__(self, c_list, w_list, p_c=0.2, p_w=0.8):
        # self.num_networks = len(c_list)
        self.n = np.shape(c_list[0])[0]
        self.c_list = c_list
        self.w_list = w_list
        self.p_c = p_c
        self.p_w = p_w

    def generate_w_n2n(self):
        w_n2n_list = []
        for w in self.w_list:
            ODi = np.sum(w, axis=1) + np.sum(w, axis=0)
            w_n2n = np.zeros((self.n, self.n))

            for i in range(self.n):
                neighbors_sort = [(w[i, j] + w[i, j], j) for j in range(self.n)]
                neighbors_sort.sort(reverse=True)

                total_flows = 0
                for (flows, node) in neighbors_sort:
                    if total_flows <= self.p_w * ODi[i]:
                        w_n2n[i, node] = flows
                        w_n2n[node, i] = flows
                        total_flows += flows

                    else:
                        break

            # weighted
            d = np.squeeze(w_n2n.sum(axis=1))
            # print(d.shape)
            D_ = np.diag(np.power(d, -1))  # (n,n)
            w_n2n_weighted = np.dot(D_, w_n2n)

            w_n2n_list.append(w_n2n_weighted)

        w_n2nsum = np.vstack(tuple(w_n2n_list))

        return w_n2n_list, w_n2nsum

    def generate_c_n2n(self):
        c_n2n_list = []
        num_c_neighbors = int(self.n * self.p_c)
        for c in self.c_list:
            c_n2n = np.zeros((self.n, self.n))

            for i in range(self.n):
                neighbors_sort = [(c[i, j], j) for j in range(self.n) if j != i]
                neighbors_sort.sort()
                for (cost, node) in neighbors_sort[:num_c_neighbors]:
                    c_n2n[i, node] = 1 / (cost + self.EPSILON)
                    c_n2n[node, i] = 1 / (cost + self.EPSILON)

            # weighted
            d = np.squeeze(c_n2n.sum(axis=1))
            # print(d.shape)
            D_ = np.diag(np.power(d, -1))  # (n,n)
            c_n2n_weighted = np.dot(D_, c_n2n)

            c_n2n_list.append(c_n2n_weighted)

        c_n2nsum = np.vstack(tuple(c_n2n_list))

        return c_n2n_list, c_n2nsum


class Labels:
    def __init__(self, c_list):
        self.num_networks = len(c_list)
        self.n = np.shape(c_list[0])[0]

    def generate_labels(self, hubs_list_DIFF, h_max):  # the list of hubs_list under different alpha and p

        labels_list = []
        hn = sum(range(2, h_max + 1))
        N_hubs = (8 - 2 + 1) * hn  # apha: 0.2-0.8; p:2-5
        for n_ in range(self.num_networks):
            labels = np.zeros((self.n, 1))

            for hubs_list in hubs_list_DIFF:
                hubs = hubs_list[n_]
                for h in hubs:
                    labels[h, 0] += 1

            labels_list.append(labels)

        labels_array = np.vstack(tuple(labels_list))
        labels_Hfrequency = labels_array / N_hubs
        return labels_Hfrequency

    # For each network, generate 5*n paris of nodes
    def pair_ids_src(self, batchsize):
        ids_src = []
        offset = 0
        for _ in range(batchsize):
            tmp = (
                np.hstack(
                    (
                        np.array(range(self.n)),
                        np.random.choice(range(self.n), size=4 * self.n, replace=True),
                    )
                )
                + offset
            )
            ids_src.append(tmp)
            offset += self.n

        return np.hstack(ids_src)

    def pair_ids_tgt(self, batchsize):
        ids_tgt = []
        offset = 0
        for _ in range(batchsize):
            tmp = (
                np.hstack(  
                    (
                        np.roll(np.arange(self.n), -1),  
                        np.random.choice(range(self.n), size=4 * self.n, replace=True),
                    )
                )
                + offset
            )
            ids_tgt.append(tmp)

            offset += self.n

        return np.hstack(ids_tgt)


def convert_sparse_to_tensor(matrix):
    # matrix = sp.coo_matrix(matrix)
    rowIndex = matrix.row
    colIndex = matrix.col
    data = matrix.data
    rowNum = matrix.shape[0]
    colNum = matrix.shape[1]
    indices = np.mat([rowIndex, colIndex]).transpose()
    return tf.SparseTensorValue(indices, data, (rowNum, colNum))


def rank_top_k(n, k, real, predict):
    if k < 1:
        TopK = max(n * k, 1)
    else:
        TopK = k

    real_inx = np.argsort(-real, axis=0)
    pre_inx = np.argsort(-predict, axis=0)
    real_topK = real_inx[0:TopK, 0]  
    predict_topK = pre_inx[0:TopK, 0] 

    accuracy = 0
    for node in predict_topK:
        if node in real_topK:
            accuracy += 1

    return accuracy / TopK
