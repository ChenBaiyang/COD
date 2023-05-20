import itertools as its
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import minmax_scale
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def relation_matrix_torch(vec1, vec2, e):
    dist_matrix = torch.cdist(vec1, vec2, p=1)
    if e == -1:
        return dist_matrix < 1e-6
    lamb = dist_matrix.mean() * e
    relation_matrix = 1 - dist_matrix
    relation_matrix[dist_matrix > lamb] = 0
    return relation_matrix

class COD(object):
    def __init__(self):
        pass

    def __make_cardinality__(self, X):
        n, m, _ = X.shape
        X = torch.tensor(X, dtype=torch.float32)
        M_R_c = torch.zeros((m, n, n), dtype=torch.float32)
        for j in range(m):
            M_R_c[j] = relation_matrix_torch(X[:, j], X[:, j], self.lambs[j])
        card1 = M_R_c.sum(dim=1) / n

        card2 = torch.zeros((m * (m - 1) // 2, n), dtype=torch.float32)
        for idx, l in enumerate(its.combinations(range(m), 2)):
            M_R_B = M_R_c[list(l)]
            temp = torch.min(M_R_B, dim=0)[0]
            card2[idx] = temp.sum(dim=1) / n
        return np.concatenate([card1, card2])

    def fit_gamma(self, X, train_y, alpha=1, pruning=None, threshold=0.1, batch=False):
        """to be uploaded later"""
        pass

    def fit(self, train_X, train_y, nominals, alpha=1):
        m = len(nominals)
        results = []
        gammas = []
        lambdas = (0.1 * np.float_power(1.5, np.arange(10))).round(3)
        for lamb in lambdas:
            self.lambs = np.full(m, lamb)
            self.lambs[nominals] = -1
            self.fit_gamma(train_X, train_y, alpha=alpha)
            gammas.append(self.gamma)
            results.append(self.train_auc)
            if nominals.sum() == m:
                break

        gammas = np.array(gammas)
        results = np.array(results)
        best_auc = np.max(results)
        best_idxs = np.where(results > best_auc - 1e-6)[0]
        best_idx = best_idxs[(len(best_idxs)-1) // 2]
        self.lamb = lambdas[best_idx]
        self.gamma = gammas[best_idx]
        self.lambs = np.full(m, self.lamb)
        self.lambs[nominals] = -1
        return self

    def predict_score(self, X = None):
        card = self.__make_cardinality__(X)
        OD = ((1 - card) * np.expand_dims(self.gamma, axis=1)).mean(axis=0)
        return OD
