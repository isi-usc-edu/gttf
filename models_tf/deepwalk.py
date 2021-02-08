
import numpy as np
import tensorflow as tf

from utils.common import asym_dot


def calculate_deepwalk_loss_on_forest(Z, forest, negative_samples=5):
  """

  Args:
    Z: embedding dictionary tensor
    forest: list of 2D int tensors of node IDs
  
  Return:
    tuple of two float (pos_loss, neg_loss) losses approximating -log P(u|v).
    where P(u|v) = softmax <L_u, R_v>
    where L_u is left-half(Z[u]) and R_v is right-half(Z[v])
    Since -log P = -log softmax = - <L_u, R_v> + LogSumExp_t <L_u, L_t>
    the returned result will contain a pair: 
      pos_loss = - <L_u, R_v>
      neg_loss =~ LogSumExp_t <L_u, L_t> [only an approximation, using `negative_samples` negatives]
  """
  num_nodes = int(Z.shape[0])
  center_embeds = tf.nn.embedding_lookup(Z, forest[0][:, 0])

  ## POSITIVES.
  mean_context_embeds = None
  window_size = len(forest) - 1
  total_weight = 0
  for w, context_nodes in enumerate(forest[1:]):
    context_embeds = tf.nn.embedding_lookup(Z, context_nodes)
    if mean_context_embeds is None:
      mean_context_embeds = tf.reduce_mean(context_embeds, axis=1) * (window_size - w) / window_size
    else:
      mean_context_embeds += tf.reduce_mean(context_embeds, axis=1) * (window_size - w) / window_size
    total_weight += (window_size - w) / window_size
  
  ## NEGATIVES.
  negs = np.random.choice(num_nodes, size=(len(forest[0]), negative_samples))
  neg_embeddings = tf.nn.embedding_lookup(Z, negs)
  # shape: num pos X num neg
  #center_dot_negs = tf.reduce_sum(tf.expand_dims(center_embeds, 1) * neg_embeddings, axis=-1)
  center_dot_negs = asym_dot(
      tf.expand_dims(center_embeds, 2),
      tf.stop_gradient(tf.transpose(neg_embeddings, (0, 2, 1))), sum_fn=tf.math.reduce_sum, axis=1)
  #center_dot_context = tf.reduce_sum(center_embeds * mean_context_embeds, axis=-1)
  center_dot_context = asym_dot(center_embeds, mean_context_embeds, sum_fn=tf.math.reduce_sum, axis=1)
  neg_loss = tf.math.reduce_logsumexp(center_dot_negs, axis=1)
  pos_loss = -center_dot_context

  return pos_loss, neg_loss
