"""
This file contains accummulation functions for GTTF

Accumulation functions can return anything but common examples include:
SampledAdjacency (for easy use with many message passing algorithms)
Loss Values (frequently used when the loss can easily be written as a function of the traversal)
"""
import numpy as np
import scipy.sparse
import copy

class SampledAdjacency:
  """Class for handling a sampled adjacency

  Most easily instantiated with `from_walk_forest` static method.

  Contains methods for trimming/untrimming adjacency and feature matrices across many frameworks.
  This offers a significant factor of improvement in most cases.
  """

  def __init__(self, edges, dense_shape):
    self.edges = edges
    self.dense_shape = dense_shape

    self.lazy_cache = {}
    """
    self._sampled_nodes = None  # Set on sampled_nodes()
    self._sp_csr = None
    self._trimmed_csr = None
    """
  def __getattr__(self, attr):
    if attr.startswith('_lazy_'):
      raise ValueError('Member not found: .' + attr[6:])
    if attr in self.lazy_cache:
      return self.lazy_cache[attr]
    method = '_lazy_' + attr
    method = getattr(self, method)
    result = method()
    self.lazy_cache[attr] = result
    return result
  
  @staticmethod
  def from_walk_forest(walk_forest, dense_shape, directed=False):
    """Used to instantiate SampledAdjacency from a walk forest"""
    edges = accumulate_unique_edges(walk_forest, directed=directed)
    return SampledAdjacency(edges, dense_shape)

  def _lazy_sp_csr(self):
    data_arr = np.ones([self.edges.shape[0]], dtype='float32')
    return scipy.sparse.csr_matrix(
        (data_arr, (self.edges[:, 0], self.edges[:, 1])),
        shape=self.dense_shape)

  def _lazy_sparse_tf(self):
    pass

  def _lazy_sampled_nodes(self):
    return np.unique(self.edges)
    #if self._sampled_nodes is None:
    #  self._sampled_nodes = 
    #return self._sampled_nodes

  def _lazy__torch(self):
    import torch
    return torch

  def _lazy_torch_trimmed(self):
    sampled_nodes = self.sampled_nodes
    rows, cols = self.csr_trimmed.nonzero()
    torch = self._torch
    #import IPython; IPython.embed()
    indices = torch.stack([
        torch.from_numpy(np.array(rows, dtype='int64')),
        torch.from_numpy(np.array(cols, dtype='int64')),
    ], axis=0)
    values = torch.ones(rows.shape)
    return torch.sparse.FloatTensor(
        indices, values,
        torch.Size([sampled_nodes.shape[0], sampled_nodes.shape[0]]))

  def _lazy_csr_trimmed(self):
    sampled_nodes = self.sampled_nodes
    return self.sp_csr[sampled_nodes][:, sampled_nodes]

  def _lazy_tf_trimmed(self):
    sampled_nodes = self.sampled_nodes
    return csr_binary_matrix_to_tf(self.csr_trimmed.nonzero())
    
  def _lazy__tf(self):
    import tensorflow
    return tensorflow
  
  def torch_trim_x(self, x, unused_x_ids=None):
    """Trims an X matrix. Keeps only the a row or a column in adjacency is set.

    Args:
      x: tensorflow dense matrix with height equals number of nodes.
      x_ids: If given, must be a vector with size == x.shape[0].
        If so, it must contain node IDs corresponding to rows in x.
        If not given, it is equivalent to passing arange(x.shape[0])
    """
    #TODO: Deal with x_ids/unused_x_ids in doc/paramaters
    return x[self.sampled_nodes]

  def torch_untrim_gather(self, y, y_ids=None):
    sampled_nodes = self.sampled_nodes
    positions = np.searchsorted(sampled_nodes, y_ids)
    return y[positions]

  def tf_trim_x(self, x, unused_x_ids=None):
    """Trims an X matrix. Keeps only the a row or a column in adjacency is set.

    Args:
      x: tensorflow dense matrix with height equals number of nodes.
      x_ids: If given, must be a vector with size == x.shape[0].
        If so, it must contain node IDs corresponding to rows in x.
        If not given, it is equivalent to passing arange(x.shape[0])
    """
    #TODO: Deal with x_ids/unused_x_ids in doc/paramaters
    sampled_nodes = self.sampled_nodes
    return self._tf.gather(x, sampled_nodes)

  def tf_untrim_gather(self, y, y_ids=None):
    sampled_nodes = self.sampled_nodes
    positions = np.searchsorted(sampled_nodes, y_ids)
    return self._tf.gather(y, positions)

def accumulate_unique_edges(walk_forest, directed=False):
  all_edges = []
  for i in range(len(walk_forest) - 1):
    src = walk_forest[i]
    dst = walk_forest[i+1]
    fanout = dst.shape[1] // src.shape[1]
    src_repeated = np.repeat(src, fanout, axis=1)
    step_edges = np.stack([src_repeated, dst], -1).reshape((-1, 2))
    all_edges.append(step_edges)
  all_edges = np.concatenate(all_edges, 0)
  all_edges = np.unique(all_edges, axis=0)

  if not directed:
    all_edges = np.concatenate([all_edges, all_edges[:, ::-1]], 0)
    all_edges = np.unique(all_edges, axis=0)

  return all_edges


def forest_to_loss_accumulation(walk_forest):
  pass



def csr_binary_matrix_to_tf(mat):
  import tensorflow as tf
  rows, cols = mat.nonzero()
  indices = tf.stack([
      tf.convert_to_tensor(rows, dtype=tf.int64),
      tf.convert_to_tensor(cols, dtype=tf.int64),
  ], axis=-1)
  values = tf.ones(rows.shape, dtype=tf.float32)
  return tf.sparse.SparseTensor(indices, values,
                                dense_shape=(mat.shape[0], mat.shape[0]))


# NOT USED YET


class AccFunction:
  """
  Class for wrapping accumulation functions

  To use...

  @AccFunction.wrap()
  def my_accumulation_function(path, v, *args, **kwargs)

  """
  def __init__(self, acc_fn, acc_arg, acc_handler):
      self.acc_fn = acc_fn
      self.acc_arg_default = copy.deepcopy(acc_arg)
      self.acc_arg = acc_arg
      self.acc_handler = acc_handler

  @staticmethod
  def wrap(acc_arg=None, acc_handler=None):
      def wrapper(wrapped):
          runnable = AccFunction(wrapped, copy.deepcopy(acc_arg), acc_handler)
          return runnable
      return wrapper

  def reset_acc(self):
      self.acc_arg = copy.deepcopy(self.acc_arg_default)

  def __call__(self, walk_forest, nodes, *args, **kwargs):
      if self.acc_arg is not None:
          return self.acc_fn(walk_forest, nodes, self.acc_arg, *args, **kwargs)
      else:
          return self.acc_fn(walk_forest, nodes, *args, **kwargs)
