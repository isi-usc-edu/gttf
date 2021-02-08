"""Script for training node embeddings via skipgram models.

This script is programmed around datasets in:
https://github.com/google/asymproj_edge_dnn/tree/master/datasets
"""

# 87
# python3.8 train_tf_skipgram.py --dataset_dir=~/data/asymproj/datasets/ca-HepTh --steps=10000 --lr 0.01 --d 128 --neg_samples=20 --fanout=2x3x4 --batch_size=1000

#89.
#python3.8 train_tf_skipgram.py --dataset_dir=~/data/asymproj/datasets/ca-HepTh --steps=10000 --lr 0.01 --d 128 --neg_samples=20 --fanout=1x1x1x1x1x1x1 --batch_size=1000

import random

from absl import flags, app

import numpy as np
import scipy.sparse as sp
from sklearn import metrics
import tensorflow as tf
import tqdm
import torch



from data import asymproj_datasets
from framework import accumulation, compact_adj, traversals
from models_tf import deepwalk
from utils.common import IdBatcher, deepwalk_eval, sym_dot

flags.DEFINE_string(
    'dataset_dir', None,
    'Directory where all dataset files live. All data files '
    'must be located here. Including {train,test}.txt.npy and '
    '{train,test}.neg.txt.npy. ')

flags.DEFINE_integer(
    'eval_test_edges', None,
    'If set, this many test edges will be evaluated.')

flags.DEFINE_string(
    'train_dir', None,
    'If set, trained models will be saved here.')

flags.DEFINE_integer('d', 16, 'Embedding size.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')

flags.DEFINE_integer('steps', 100, 'Number of steps')
flags.DEFINE_integer('neg_samples', 5, 'Number negatives per positive')
flags.DEFINE_integer('batch_size', 100, 'Number of nodes in batch')
flags.DEFINE_string('fanout', 'x'.join(['1']*10), 'Number of nodes in batch')
flags.DEFINE_float('pos_coeff', 5.0, 'Coeffiecient for positive part of the objective.')

FLAGS = flags.FLAGS


def main(_):
  num_nodes, train_edges, test_pos_arr, test_neg_arr, is_directed = asymproj_datasets.read_dataset(FLAGS.dataset_dir)
  adj_rows = train_edges[:, 0]
  adj_cols = train_edges[:, 1]
  if not is_directed:
    adj_rows = np.concatenate([adj_rows, train_edges[:, 1]], axis=0)
    adj_cols = np.concatenate([adj_cols, train_edges[:, 0]], axis=0)

  csr_adj = sp.csr_matrix(
      (np.ones(len(adj_rows), dtype='int32'),  (adj_rows, adj_cols)),
      shape=(num_nodes, num_nodes))
  cadj = compact_adj.CompactAdjacency(csr_adj)

  # Trainable embeddings.
  Z = tf.Variable(np.random.uniform(low=-0.01, high=0.01, size=[num_nodes, FLAGS.d]), dtype=tf.float32, name='Y')
  lr = tf.Variable(FLAGS.lr, name='LR')

  opt = tf.keras.optimizers.Adam(lr, clipnorm=0.01)

  batcher = IdBatcher(set(adj_rows))
  fanouts = [int(f) for f in FLAGS.fanout.split('x')]
  tt = tqdm.tqdm(range(FLAGS.steps))
  for i in tt:
    seed_ids = batcher.batch(FLAGS.batch_size)
    seed_ids = np.array(seed_ids)
    forest = traversals.np_traverse(cadj, seed_ids, fanouts)
    
    with tf.GradientTape() as tape:
      pos_loss, neg_loss = deepwalk.calculate_deepwalk_loss_on_forest(Z, forest, negative_samples=FLAGS.neg_samples)
      total_loss = FLAGS.pos_coeff * pos_loss + neg_loss

    gZ = tape.gradient(total_loss, Z)
    opt.apply_gradients([(gZ, Z)])

    ## Eval
    npe = Z.numpy()
    # test_accuracy = deepwalk_eval(npe, test_pos_arr, test_neg_arr)
    test_scores = sym_dot(npe[test_pos_arr[:, 0]],  npe[test_pos_arr[:, 1]], np.sum)
    test_neg_scores = sym_dot(npe[test_neg_arr[:, 0]], npe[test_neg_arr[:, 1]], np.sum)
    test_y = [0] * len(test_neg_scores) + [1] * len(test_scores)
    test_y_pred = np.concatenate([test_neg_scores, test_scores], 0)
    test_accuracy = metrics.roc_auc_score(test_y, test_y_pred)
    tt.set_description('Test=%g' % test_accuracy)
    



if __name__ == '__main__':
  app.run(main)