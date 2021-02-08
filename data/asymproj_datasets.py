import os
import pickle

import numpy as np

def read_dataset(dataset_dir):
  """Reads graph dataset from directory.

  These datasets are used in WatchYourStep and AsymProjection, available on:
  https://github.com/google/asymproj_edge_dnn/tree/master/datasets

  Args:
    dataset_dir: directory of dataset containing files {train, test, test.neg}.txt.npy and index.pkl

  Returns:
    tuple (num_nodes, train_edges, test_pos_arr, test_neg_arr)
  """
  dataset_dir = os.path.expanduser(dataset_dir)
  
  ## Load train edges
  if os.path.exists(os.path.join(dataset_dir, 'train_c.txt.npz')):
    print('Loading taking compressed version.')
    train_edges = np.load(
      open(os.path.join(dataset_dir, 'train_c.txt.npz'), 'rb'))['arr_0']
  else:
    train_edges = np.load(
      open(os.path.join(dataset_dir, 'train.txt.npy'), 'rb'))

  ## Load test edges.
  if os.path.exists(os.path.join(dataset_dir, 'test_c.txt.npz')):
    test_pos_file = os.path.join(dataset_dir, 'test_c.txt.npz')
    test_pos_arr = np.load(open(test_pos_file, 'rb'))['arr_0']
  else:
    test_pos_file = os.path.join(dataset_dir, 'test.txt.npy')
    test_pos_arr = np.load(open(test_pos_file, 'rb'))

  index_file = os.path.join(dataset_dir, 'index.pkl')
  if os.path.exists(index_file):
    index = pickle.load(open(index_file, 'rb'))
    if 'index' in index:
      num_nodes = len(index['index'])
    else:
      num_nodes = len(index)
  else:
    num_nodes = train_edges.max() + 1
  directed_negs_filename = os.path.join(dataset_dir, 'test.directed.neg.txt.npy')
  is_directed = os.path.exists(directed_negs_filename)
  if is_directed:
    test_neg_arr = np.load(open(directed_negs_filename, 'rb'))
  else:
    test_neg_file = os.path.join(dataset_dir, 'test.neg.txt.npy')
    test_neg_arr = np.load(open(test_neg_file, 'rb'))

  return num_nodes, train_edges, test_pos_arr, test_neg_arr, is_directed

