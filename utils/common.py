import torch
import random
import numpy as np
import tensorflow as tf

from sklearn import metrics


class IdBatcher:

    def __init__(self, ids):
        self.ids = list(ids)
        self.queue = []

    def batch(self, batch_size):
        while len(self.queue) < batch_size:
            shuffled_ids = [ii for ii in self.ids]
            random.shuffle(shuffled_ids)
            self.queue += shuffled_ids

        selected_batch = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        return selected_batch


def sym_dot(embeds1, embeds2, sum_fn=np.sum):
    return sum_fn(embeds1 * embeds2, axis=1)


def asym_dot(embeds1, embeds2, sum_fn, **kwargs):
  size = embeds1.shape[1]
  hs = size//2
  return sum_fn(embeds1[:, :hs] * embeds2[:, hs:], **kwargs)


def skipgram_eval(embed1, embed2, test_pos, test_neg, directed=False):
    if directed:
        test_scores = sym_dot(embed1[test_pos[:, 0]], embed2[test_pos[:, 1]], np.sum)
        test_neg_scores = sym_dot(embed1[test_neg[:, 0]], embed2[test_neg[:, 1]], np.sum)
    else:
        test_scores = 0.5*sym_dot(embed1[test_pos[:, 0]], embed2[test_pos[:, 1]], np.sum) + 0.5*sym_dot(embed2[test_pos[:, 0]], embed1[test_pos[:, 1]], np.sum)
        test_neg_scores = 0.5*sym_dot(embed1[test_neg[:, 0]], embed2[test_neg[:, 1]], np.sum) + 0.5*sym_dot(embed2[test_neg[:, 0]], embed1[test_neg[:, 1]], np.sum)
    test_y = [0] * len(test_neg_scores) + [1] * len(test_scores)
    test_y_pred = np.concatenate([test_neg_scores, test_scores], 0)
    test_accuracy = metrics.roc_auc_score(test_y, test_y_pred)
    
    return test_accuracy

