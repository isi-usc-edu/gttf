
import collections
import os
import pickle

import numpy as np
import scipy.sparse
from framework.compact_adj import CompactAdjacency as CompAdj


def concatenate_csr_matrices_by_rows(matrix1, matrix2):
  """Concatenates sparse csr matrices matrix1 above matrix2.
  
  Adapted from:
  https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
  """
  new_data = np.concatenate((matrix1.data, matrix2.data))
  new_indices = np.concatenate((matrix1.indices, matrix2.indices))
  new_ind_ptr = matrix2.indptr + len(matrix1.data)
  new_ind_ptr = new_ind_ptr[1:]
  new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

  return scipy.sparse.csr_matrix((new_data, new_indices, new_ind_ptr))



def load_x(filename):
  return pickle.load(open(filename, 'rb'), encoding='latin1')

def read_planetoid_dataset(dataset_name='ind.cora', dataset_dir='~/data/planetoid/data/'):
  base_path = os.path.expanduser(os.path.join(dataset_dir, dataset_name))
  if not os.path.exists(os.path.expanduser(dataset_dir)):
    raise ValueError('cannot find dataset_dir=%s. Please:\nmkdir -p ~/data; cd ~/data; git clone git@github.com:kimiyoung/planetoid.git')
  edge_lists = pickle.load(open(base_path + '.graph', 'rb'))

  allx = load_x(base_path + '.allx')
  
  ally = np.array(np.load(base_path + '.ally', allow_pickle=True), dtype='float32')
 
  testx = load_x(base_path + '.tx')

  # Add test
  test_idx = list(map(int, open(base_path + '.test.index').read().split('\n')[:-1]))

  num_test_examples = max(test_idx) - min(test_idx) + 1
  sparse_zeros = scipy.sparse.csr_matrix((num_test_examples, allx.shape[1]),
                                         dtype='float32')

  allx = concatenate_csr_matrices_by_rows(allx, sparse_zeros)
  llallx = allx.tolil()
  llallx[test_idx] = testx
  #allx = scipy.vstack([allx, sparse_zeros])

  test_idx_set = set(test_idx)


  testy = np.array(np.load(base_path + '.ty', allow_pickle=True), dtype='float32')
  ally = np.concatenate(
      [ally, np.zeros((num_test_examples, ally.shape[1]), dtype='float32')],
      0)
  ally[test_idx] = testy

  num_nodes = len(edge_lists)

  # Will be used to construct (sparse) adjacency matrix.
  edge_sets = collections.defaultdict(set)
  for node, neighbors in edge_lists.items():
    for n in neighbors:
      edge_sets[node].add(n)
      edge_sets[n].add(node)  # Assume undirected.

  # Now, build adjacency list.
  adj_indices = []
  adj_values = []
  for node, neighbors in edge_sets.items():
    for n in neighbors:
      adj_indices.append((node, n))
      adj_values.append(1)

  adj_indices = np.array(adj_indices, dtype='int32')
  adj_values = np.array(adj_values, dtype='int32')
  
  adj = scipy.sparse.csr_matrix((num_nodes, num_nodes), dtype='int32')

  adj[adj_indices[:, 0], adj_indices[:, 1]] = adj_values

  return adj, llallx, ally, test_idx

def load_reddit(path):
    train_comp_adj = CompAdj.from_file(path + '/train_comp_adj_reddit.pkl')
    test_comp_adj = CompAdj.from_file(path + '/test_comp_adj_reddit.pkl')
    feat_data = np.load(path + '/feat_data.npy')
    labels = np.load(path + '/labels.npy')
    train_ids = np.load(path + '/train_ids.npy')
    val_ids = np.load(path + '/val_ids.npy')
    test_ids = np.load(path + '/test_ids.npy')

    return train_comp_adj, test_comp_adj, feat_data, labels, train_ids, val_ids, test_ids


def load_amazon(path):
  comp_adj = CompAdj.from_file(path + '/comp_adj.pkl')
  train_comp_adj = CompAdj.from_file(path + '/train_comp_adj.pkl')
  feat_data = np.load(path + '/feat_data.npy')
  labels = np.load(path + '/labels.npy')
  train_ids = np.load(path + '/train_ids.npy')
  val_ids = np.load(path + '/val_ids.npy')
  test_ids = np.load(path + '/test_ids.npy')

  return feat_data, labels, train_comp_adj, comp_adj, train_ids, val_ids, test_ids


def load_ogbproducts(path):
  comp_adj = CompAdj.from_file(path + '/products_comp_adj.pkl')
  degrees = np.load(path + '/degrees.npy')
  feat_data = np.load(path + '/feat_data.npy')
  labels = np.load(path + '/labels.npy')
  train_ids = np.load(path + '/train_ids.npy')
  val_ids = np.load(path + '/val_ids.npy')
  test_ids = np.load(path + '/test_ids.npy')

  return feat_data, labels, comp_adj, degrees, train_ids, val_ids, test_ids