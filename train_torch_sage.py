import sys
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np
import os
import scipy.sparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from absl import flags, app

from models_pytorch.sage import GraphSage

from framework.compact_adj import CompactAdjacency as CompAdj
from framework import traversals
from training_loops import datasets


from tqdm import tqdm

flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epochs', 200, '')
flags.DEFINE_integer('batch_size', 512, '')
flags.DEFINE_integer('num_workers', 0, 'Number of dataloader workers to spawn during training')
flags.DEFINE_string('fanouts', '5x5', 'list of integers separated with "x".')
flags.DEFINE_string('device', 'cpu', 'Should be "cpu" or "cuda" or any valid torch device name')
flags.DEFINE_string('data_dir', '../../datasets', 'Data directory')
flags.DEFINE_string('dataset', 'reddit', 'Dataset')
flags.DEFINE_string('hidden_dims', '256,256', 'Comma separated list of hidden dims')
flags.DEFINE_float('dropout', 0.5, 'Dropout probability')

FLAGS = flags.FLAGS


class GraphDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, V, labels):
        self.V = V
        self.labels = labels

    def __getitem__(self, index):
        return self.V[index], self.labels[self.V[index]]

    def __len__(self):
        return len(self.V)


class WalkForestCollator:
    def __init__(self, compAdj: CompAdj, fanouts):
        self.compAdj = compAdj
        self.fanouts = fanouts

    def __call__(self, batch):
        batch_np = np.array(batch, dtype=object)
        node_ids = batch_np[:, 0].astype(int)
        node_labels = torch.from_numpy(batch_np[:, 1].astype(int).flatten())
        forest = traversals.np_traverse(self.compAdj, node_ids, self.fanouts)
        torch_forest = [torch.from_numpy(forest[0]).flatten()]
        for i in range(len(forest) - 1):
            torch_forest.append(torch.from_numpy(forest[i + 1]).reshape(-1, self.fanouts[i]))

        return torch_forest, node_labels


def train(feat_data, labels, train_comp_adj, val_comp_adj, train_ids, val_ids, fanouts):
    dev = torch.device("cuda:0" if (torch.cuda.is_available() & (FLAGS.device == 'cuda')) else "cpu")
    print('Training on {}'.format(str(dev)))

    train_collator = WalkForestCollator(train_comp_adj, fanouts)
    val_collator = WalkForestCollator(val_comp_adj, fanouts)

    train_dataset = GraphDataset(train_ids, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers,
                                               collate_fn=train_collator, pin_memory=True)

    val_dataset = GraphDataset(val_ids, labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=True, num_workers=FLAGS.num_workers,
                                             collate_fn=val_collator, pin_memory=True)

    # Normalize features
    train_feats = feat_data[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feat_data = scaler.transform(feat_data)

    hidden_dims = list(map(lambda x: int(x), FLAGS.hidden_dims.split(',')))
    assert len(hidden_dims) == 2, 'Must specify only two hidden dims for a 2 layer SAGE model'

    num_classes = np.max(labels) + 1
    net = GraphSage(feat_data, num_classes, hidden_dims[0], hidden_dims[1], FLAGS.dropout)
    net.to(dev)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=FLAGS.lr)

    start = timer()
    for epoch in range(FLAGS.epochs):
        net.train()
        Y_true = []
        Y_pred = []
        for i, (batch_forest, batch_labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            forest = [level.to(device=dev, dtype=torch.long, non_blocking=True) for level in batch_forest]
            batch_labels = batch_labels.to(dev, non_blocking=True).long()

            scores = net(forest)

            loss = loss_fn(scores, batch_labels)
            loss.backward()
            optimizer.step()

    end = timer()

    net.eval()
    with torch.no_grad():
        for i, (batch_forest, batch_labels) in enumerate(tqdm(val_loader)):
            forest = [level.to(device=dev, dtype=torch.long, non_blocking=True) for level in batch_forest]
            batch_labels = batch_labels.to(dev, non_blocking=True)

            scores = net(forest)
            y_pred = scores.cpu().data.numpy().argmax(axis=1)
            y_true = batch_labels.cpu().numpy()
            Y_true += list(y_true)
            Y_pred += list(y_pred)

        f1 = f1_score(Y_true, Y_pred, average="micro")
        print('Test F1 = {}'.format(f1))

    print('Training time = {} seconds'.format(end - start))


def eval_acc(y_true, y_pred):
    total = len(y_true)
    correct = (y_true == y_pred).sum()

    return correct / total


def main(_):
    path = os.path.join(FLAGS.data_dir, FLAGS.dataset)
    fanouts = list(map(lambda x: int(x), FLAGS.fanouts.split('x')))

    if(FLAGS.dataset == 'reddit'):
        train_comp_adj, test_comp_adj, feat_data, labels, train_ids, val_ids, test_ids = datasets.load_reddit(path)
        train(feat_data, labels, train_comp_adj, test_comp_adj, train_ids, test_ids, fanouts)

    elif(FLAGS.dataset == 'products'):
        feat_data, labels, comp_adj, degrees, train_ids, val_ids, test_ids = datasets.load_ogbproducts(path)
        train(feat_data, labels, comp_adj, comp_adj, train_ids, test_ids, fanouts)

    elif(FLAGS.dataset == 'amazon'):
        feat_data, labels, train_comp_adj, comp_adj, train_ids, val_ids, test_ids = datasets.load_amazon(path)
        train(feat_data, labels, train_comp_adj, comp_adj, train_ids, test_ids, fanouts)


if __name__ == '__main__':
    app.run(main)
