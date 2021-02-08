import tensorflow as tf


def kipf_renorm_tf(adj):
  # adj: a tf sparse tensor, possibly on gpu
  # Kipf's re-norm trick
  adj = tf.sparse.maximum(adj, tf.sparse.transpose(adj))  # Make symmetric
  
  num_nodes = adj.shape[0]
  
  eye = tf.sparse.eye(num_nodes, dtype='float32')
  adj = tf.sparse.maximum(adj, eye) # Set diagonal

  sums = tf.sparse.reduce_sum(adj, 1)
  inv_sqrt_degree = tf.sqrt(1.0/sums)
  # convert infinities to zeros
  #inv_sqrt_degree = tf.math.multiply_no_nan(inv_sqrt_degree, tf.cast(inv_sqrt_degree < 2, tf.float32))
  normed_adj = adj * inv_sqrt_degree * tf.reshape(inv_sqrt_degree, (num_nodes,1))
  return normed_adj



def adj_times_x(adj, x, adj_pow=1):
  """Multiplies (adj^adj_pow)*x."""
  for i in range(adj_pow):
    x =  tf.sparse.sparse_dense_matmul(adj, x)
  return x




