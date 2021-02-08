"""
Script copied from https://github.com/chennnM/GCNII

GCNII can be run using train_torch_gcn but does not fully reproduce 
the same results without copying the GCNII pre-processing steps (as done here).
"""

from __future__ import division
from __future__ import print_function
import os
import sys
import time
import random
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import torch
import torch.nn.functional as F
import torch.optim as optim
import uuid
import scipy.sparse as sp

from utils import torch_utils
from models_pytorch.gcnii import GCNII
from framework import accumulation, compact_adj, traversals

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--fanouts', default="10x5x5", help="Fanouts for GTTF (e.g. 5x5x5).")
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.fanouts = [int(f) for f in args.fanouts.split('x')]

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# adapted from tkipf/gcn
def load_citation(dataset_str="cora", dataset_dir="~/data/planetoid/data"):
    """
    Load Citation Networks Datasets.
    """
    basepath = os.path.expanduser(os.path.join(dataset_dir))
    if not os.path.exists(os.path.expanduser(dataset_dir)):
        raise ValueError('cannot find dataset_dir=%s. Please:\nmkdir -p ~/data; cd ~/data; git clone git@github.com:kimiyoung/planetoid.git')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(basepath, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(basepath, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test

def run_model_with_gttf(model, idx):
    forest = traversals.np_traverse(c_adj, np.array(idx), fanouts=args.fanouts)
    dense_shape = (num_nodes, num_nodes)
    sampled_adj = accumulation.SampledAdjacency.from_walk_forest(forest, dense_shape)
    trimmed_adj = sampled_adj.csr_trimmed
    trimmed_adj = torch_utils.kipf_renorm_sp(trimmed_adj)
    trimmed_adj = torch_utils.sparse_mx_to_torch_sparse_tensor(trimmed_adj)
    trimmed_adj = trimmed_adj.to(device)
    trimmed_x = sampled_adj.torch_trim_x(features)
    trimmed_x = trimmed_x.to(device)
    output = model(trimmed_x,trimmed_adj)
    output = sampled_adj.torch_untrim_gather(output, np.array(idx))
    return output

# Load data
adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
num_nodes = adj.shape[0]
c_adj = compact_adj.CompactAdjacency(adj)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
checkpt_file = 'checkpoints/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

model = GCNII(nfeat=features.shape[1],
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                variant=args.variant).to(device)

optimizer = optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ],lr=args.lr)

def train():
    model.train()
    optimizer.zero_grad()
    output = run_model_with_gttf(model, idx_train)
    acc_train = accuracy(output, labels[idx_train].to(device))
    loss_train = F.nll_loss(output, labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()

def validate():
    model.eval()
    with torch.no_grad():
        output = run_model_with_gttf(model, idx_val)
        loss_val = F.nll_loss(output, labels[idx_val].to(device))
        acc_val = accuracy(output, labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = run_model_with_gttf(model, idx_test)
        loss_test = F.nll_loss(output, labels[idx_test].to(device))
        acc_test = accuracy(output, labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()

t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
for epoch in range(args.epochs):
    loss_tra,acc_tra = train()
    loss_val,acc_val = validate()
    if(epoch+1)%1 == 0: 
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            'acc:{:.2f}'.format(acc_tra*100),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'acc:{:.2f}'.format(acc_val*100))
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

if args.test:
    acc = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))
    





