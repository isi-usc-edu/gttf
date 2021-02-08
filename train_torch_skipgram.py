"""Script for training node embeddings via skipgram models.

This script is programmed around datasets in:
https://github.com/google/asymproj_edge_dnn/tree/master/datasets
"""

# 87
# python3.8 train_tf_skipgram.py --dataset_dir=~/data/asymproj/datasets/ca-HepTh --steps=10000 --lr 0.01 --d 128 --neg_samples=20 --fanout=2x3x4 --batch_size=1000

# 89.
# python3.8 train_tf_skipgram.py --dataset_dir=~/data/asymproj/datasets/ca-HepTh --steps=10000 --lr 0.01 --d 128 --neg_samples=20 --fanout=1x1x1x1x1x1x1 --batch_size=1000

import random

from absl import flags, app

import numpy as np
import scipy.sparse as sp
import torch
import tensorflow as tf
from torch import optim
import tqdm

from data import asymproj_datasets
from framework import accumulation, compact_adj, traversals
import models_pytorch
from models_pytorch import wys, deepwalk
from utils.common import IdBatcher, skipgram_eval

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
flags.DEFINE_string('fanout', 'x'.join(['1' * 10]), 'Number of nodes in batch')
flags.DEFINE_float('pos_coeff', 5.0, 'Coeffiecient for positive part of the objective.')
flags.DEFINE_string('model', 'deepwalk.DeepWalk', 'Model to use from models_pytorch')

FLAGS = flags.FLAGS


def main(_):
    num_nodes, train_edges, test_pos_arr, test_neg_arr, is_directed = asymproj_datasets.read_dataset(FLAGS.dataset_dir)
    adj_rows = train_edges[:, 0]
    adj_cols = train_edges[:, 1]
    if not is_directed:
        adj_rows = np.concatenate([adj_rows, train_edges[:, 1]], axis=0)
        adj_cols = np.concatenate([adj_cols, train_edges[:, 0]], axis=0)

    csr_adj = sp.csr_matrix(
        (np.ones(len(adj_rows), dtype='int32'), (adj_rows, adj_cols)),
        shape=(num_nodes, num_nodes))
    cadj = compact_adj.CompactAdjacency(csr_adj)

    fanouts = [int(f) for f in FLAGS.fanout.split('x')]
    model_class = models_pytorch.__dict__[FLAGS.model.split('.')[0]].__dict__[FLAGS.model.split('.')[1]]
    model = model_class(num_nodes=num_nodes, emb_dim=FLAGS.d, window_size=len(fanouts))
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FLAGS.lr)

    batcher = IdBatcher(set(adj_rows))
    tt = tqdm.tqdm(range(FLAGS.steps))
    for i in tt:
        model.train()
        opt.zero_grad()
        seed_ids = batcher.batch(FLAGS.batch_size)
        seed_ids = np.array(seed_ids)
        forest = [torch.LongTensor(x) for x in traversals.np_traverse(cadj, seed_ids, fanouts)]

        loss_train = model.loss(forest, negative_samples=FLAGS.neg_samples, pos_coeff=FLAGS.pos_coeff)

        loss_train.backward()
        opt.step()

        ## Eval
        embeddings = [e.weight.detach().numpy() for e in model.embeddings()]
        if len(embeddings) == 1:
            test_accuracy = skipgram_eval(embeddings[0], embeddings[0], test_pos_arr, test_neg_arr, directed=False)
        else:
            test_accuracy = skipgram_eval(embeddings[0], embeddings[1], test_pos_arr, test_neg_arr, directed=False)
        tt.set_description('Test=%g' % test_accuracy)


if __name__ == '__main__':
    app.run(main)