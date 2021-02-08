"""
Sampling Functions (Bias Functions)

This file contains sampling functions that adhere to the Sampler interface.

They accept a Compact Adjacency, a Walk Forest, and a fanout.
They return a batch of next neighbor ids.
"""

import numpy as np


# TODO: Add checker/testing function for future developers
# Should check that sampler accepts a compact_adj, tree, fanout
# and returns an appropriate batch of next neighbor ids


def np_uniform_sample_next(compact_adj, tree, fanout):
  "Uniform SampleFunction for unbiased walks"
  last_level = tree[-1]  # [batch, f^depth]
  batch_lengths = compact_adj.degrees[last_level]
  nodes = np.repeat(last_level, fanout, axis=1)
  batch_lengths = np.repeat(batch_lengths, fanout, axis=1)
  batch_next_neighbor_ids = np.random.uniform(size=batch_lengths.shape, low=0, high=1 - 1e-9)
  # Shape = (len(nodes), neighbors_per_node)
  batch_next_neighbor_ids = np.array(
      batch_next_neighbor_ids * batch_lengths,
      dtype=last_level.dtype)
  shape = batch_next_neighbor_ids.shape
  batch_next_neighbor_ids = np.array(compact_adj.compact_adj[nodes.reshape(-1), batch_next_neighbor_ids.reshape(-1)]).reshape(shape)

  return batch_next_neighbor_ids

#@BiasFunction.wrap()
#def np_node2vec_sample_next(compact_adj, tree, fanout, p, q):

'''

#def traverse(compact_adj, ):

class Sampler:
    def __init__(self, bias_fn, compact_adj, with_replacemnet=False):
        self.compact_adj = compact_adj
        self.bias_fn = bias_fn 
        self.sample_by_batch = self.bias_fn.sample_by_batch if self.bias_fn is not None else False
        self.with_replacement = with_replacemnet

    def sample_with_replacement(self):
        self.with_replacement = True
        return self

    def sample_neighbors(self, nodes, samples_per_node, **kwargs):
        if self.bias_fn is not None:
            return self.sample_with_bias_fn(nodes, samples_per_node, **kwargs)
        else:
            return self.sample_without_bias_fn(nodes, samples_per_node, self.with_replacement, **kwargs)

    def sample_without_bias_fn(self, nodes, samples_per_node, replace, **kwargs):
        if replace:
            batch_rep = nodes.repeat(samples_per_node)
            x = np.random.rand(nodes.shape[0] * samples_per_node)
            neighbors = np.array(self.compact_adj.compact_adj[batch_rep, np.floor(x * self.compact_adj.lengths[batch_rep]).astype(int)])			
            neighbors = neighbors.reshape(nodes.shape[0], samples_per_node)

            return neighbors
        
        neighbors = -1 * np.ones((len(nodes), samples_per_node), dtype=np.int32)

        degrees = self.compact_adj.lengths[nodes]
        thresholds = degrees <= samples_per_node
        sample_with_replacement_idxs = thresholds.nonzero()[0]
        sample_without_replacement_idxs = (thresholds == False).nonzero()[0]

        sample_with_replacement_node_ids = nodes[sample_with_replacement_idxs]
        sample_without_replacement_node_ids = nodes[sample_without_replacement_idxs]

        # With replacement
        if sample_with_replacement_idxs.shape[0] > 0:
            replacement_neighbors = self.sample_without_bias_fn(sample_with_replacement_node_ids, samples_per_node, True)
            neighbors[sample_with_replacement_idxs, :] = replacement_neighbors
		
        # Without replacement
        if sample_without_replacement_idxs.shape[0] > 0:
            degrees = self.compact_adj.lengths[sample_without_replacement_node_ids]
            max_degree = np.max(degrees)
            potential_neighbors = np.arange(max_degree)
            np.random.shuffle(potential_neighbors)
            potential_neighbors_batch = np.tile(potential_neighbors, (degrees.shape[0], 1))
            neighboring_idx_feasibility_batch = potential_neighbors_batch < degrees.reshape(-1, 1)
            #
            neighboring_arr_idxs = np.argsort((neighboring_idx_feasibility_batch == False), axis=1)[:, :samples_per_node]
            neighboring_node_idxs = potential_neighbors_batch[np.arange(degrees.shape[0]).reshape(-1, 1), neighboring_arr_idxs]
            neighboring_node_ids = self.compact_adj.compact_adj[sample_without_replacement_node_ids.reshape(-1, 1), neighboring_node_idxs].toarray()
            neighbors[sample_without_replacement_idxs, :] = neighboring_node_ids
			
        return neighbors 

    def sample_with_bias_fn(self, nodes, neighbors_per_node, **kwargs):
        if self.bias_fn is None:
            raise Exception("Cannot call sample_with_bias_fn when bias_fn is None")

        neighbors = np.zeros(shape=[len(nodes), neighbors_per_node], dtype='int32') - 1
        # NOTE: Can use tf.searchsorted(
        # sorted_sequence, values, side='left', out_type=tf.dtypes.int32, name=None)
        if self.bias_fn.sample_by_batch:
            sample_prob = self.bias_fn(nodes, neighbors=np.arange(self.compact_adj.num_nodes),
                                       compact_adj=self.compact_adj, **kwargs)
            neigbors = np.random.choice(self.compact_adj.num_nodes, size=(len(nodes), neighbors_per_node), p=sample_prob,
                                        replace=self.with_replacement)
            return neigbors

        for j, n in enumerate(nodes):
            if n == -1: continue

            sample_prob = None
            sample_prob = self.bias_fn(n, neighbors=self.compact_adj.neighbors_of(n), 
                                       compact_adj=self.compact_adj, **kwargs)
                #print('n=%i, sample_prob=%s' % (n, str(sample_prob)))
            if sample_prob is 0: continue

            num_neighbors = self.compact_adj.lengths[n]
            if not self.with_replacement and sample_prob is not None and sample_prob.nonzero()[0].shape[0] < neighbors_per_node:
                sample_number = sample_prob.nonzero()[0].shape[0]
            else:
                sample_number = neighbors_per_node
                
            # takes p.
            uniform_sample = np.random.choice(num_neighbors, 
                                              size=sample_number, 
                                              replace=self.with_replacement, p=sample_prob)
            neighbors[j, :sample_number] = np.array(self.compact_adj.compact_adj[n, uniform_sample].todense())[0]
        #import IPython; IPython.embed()
        return neighbors


class BiasFunction:
    def __init__(self, fn, sample_by_batch):
            self.sample_by_batch = sample_by_batch
            self.fn = fn

    @staticmethod
    def wrap(sample_by_batch=False):
        # should check that takes nodes and **kwargs at end
        def wrapper(wrapped):
            return BiasFunction(wrapped, sample_by_batch)
        return wrapper

    def __call__(self, nodes, **kwargs):
        return self.fn(nodes, **kwargs)


# @BiasFunction.wrap(sample_by_batch=False)
# def rooted_adj_bias(v, visited=set()):
#     return 0 if v in visited else None


# @RunnableDRW.wrap(bias_fn=rooted_adj_bias, visited=set)
# def drw_rooted_adjacency(path, v, sampled_adj, visited, **kwargs):
#   """Used for GCN, GraphSAGE, MixHop."""
#   valid_nodes = (v != -1)
#   visited.update(list(path[-1][valid_nodes]))
#   sampled_adj[path[-1][valid_nodes], v[valid_nodes]] = 1
'''