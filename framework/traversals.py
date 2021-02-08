"""
This file contains functions for conducting graph traversals.

At the moment there is just one traversal (np_traverse) for traversing according to a given sample_fn
but other functions may be developed.
"""


from . import samplers

import numpy as np


def np_traverse(compact_adj, seed_nodes, fanouts=(1,), sample_fn=samplers.np_uniform_sample_next):
  """Traverses using numpy. Returns walk tree.
  
  Args:
    compact_adj: instance of CompactAdj
    seed_nodes: np.array of type int32. Can be 1D or 2D.
    fanouts: The list (iterable) of fanout sizes to use on each step
    sample_fn: The sampling function to describe the random walk 
  """
  if not isinstance(seed_nodes, np.ndarray):
    raise ValueError('Seed must a numpy array')
  
  if len(seed_nodes.shape) > 2 or len(seed_nodes.shape) < 1 or not str(seed_nodes.dtype).startswith('int'):
    raise ValueError('seed_nodes must be 1D or 2D int array')
  
  if len(seed_nodes.shape) == 1:
    seed_nodes = np.expand_dims(seed_nodes, 1)

  # Make walk-tree
  forest_array = [seed_nodes]
  for f in fanouts:
    next_level = sample_fn(compact_adj, forest_array, f)
    assert next_level.shape[1] == forest_array[-1].shape[1]*f

    forest_array.append(next_level)

  return forest_array
  