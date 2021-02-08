import pickle
import gzip
import os

from tqdm import tqdm
import numpy as np
import scipy.sparse


# TODO: degrees is optional?
# TODO: subset: explain it.
class CompactAdjacency:

  def __init__(self, adj, precomputed=None, subset=None):
    """Constructs CompactAdjacency.

    Args:
      adj: scipy sparse matrix containing full adjacency.
      precomputed: If given, must be a tuple (compact_adj, degrees).
        In this case, adj must be None. If supplied, subset will be ignored.
    """
    if adj is None:
      return

    if precomputed:
      if adj is not None:
        raise ValueError('Both adj and precomputed are set.')
      if subset is not None:
        print('WARNING: subset is provided. It is ignored, since precomputed is supplied.')
      self.compact_adj, self.degrees = precomputed
      self.num_nodes = len(self.degrees)
    else:
      self.adj = adj
      self.num_nodes = len(self.adj) if isinstance(self.adj, dict) else self.adj.shape[0]
      self.compact_adj = scipy.sparse.dok_matrix(
          (self.num_nodes, self.num_nodes), dtype='int32')
      self.degrees = np.zeros(shape=[self.num_nodes], dtype='int32')
      self.node_set = set(subset) if subset is not None else None

      for v in tqdm(range(self.num_nodes), desc='Compacting adjacency'):
        if isinstance(self.adj, dict) and self.node_set is not None:
          connection_ids = np.array(list(self.adj[v].intersection(self.node_set)))
        elif isinstance(self.adj, dict) and self.node_set is None:
          connection_ids = np.array(list(self.adj[v]))
        else:
          connection_ids = self.adj[v].nonzero()[1]

        self.degrees[v] = len(connection_ids)
        self.compact_adj[v, np.arange(len(connection_ids), dtype='int32')] = connection_ids

    print('Converting to csr')
    self.compact_adj = self.compact_adj.tocsr()

  @staticmethod
  def from_file(filename):
    instance = CompactAdjacency(None, None)
    data = pickle.load(gzip.open(filename, 'rb'))
    instance.compact_adj = data['compact_adj']
    instance.adj = data['adj']
    instance.degrees = data['degrees'] if 'degrees' in data else data['lengths']
    instance.num_nodes = data['num_nodes']
    return instance

  @staticmethod
  def from_directory(directory):
    instance = CompactAdjacency(None, None)
    instance.degrees = np.load(os.path.join(directory, 'degrees.npy'))
    instance.compact_adj = scipy.sparse.load_npz(os.path.join(directory, 'cadj.npz'))
    print('\n\ncompact_adj.py from_directory\n\n')
    # Make adj from cadj and save to adj.npz
    import IPython; IPython.embed()
    instance.adj = scipy.sparse.load_npz(os.path.join(directory, 'adj.npz'))
    instance.num_nodes = instance.adj.shape[0]
    return instance

  def save(self, filename):
    with gzip.open(filename, 'wb') as fout:
      pickle.dump({
        'compact_adj': self.compact_adj,
        'adj': self.adj,
        'degrees': self.degrees,
        'num_nodes': self.num_nodes,
      }, fout)

  def neighbors_of(self, node):
    neighbors = self.compact_adj[node, :self.degrees[node]].todense()
    return np.array(neighbors)[0]