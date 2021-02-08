
import tensorflow as tf

from utils import tf_utils


class MixHopGCN:
  
  def __init__(self, layer_dims, use_bias=False, **kwargs):
    """
    Args:
      layer_dims: must be list of lists. Outer list indicates number of layers
        and inner list determines the dimensions devoted to each adjacency power.
    """
    self.layer_dims = layer_dims
    self.mixhop_layers = []
    self.all_layers = []
    self.max_power = 0
    for i, layer_d in enumerate(layer_dims):
      mixhop_layer = []
      self.mixhop_layers.append(mixhop_layer)
      self.max_power = max(self.max_power, len(layer_d) - 1)
      for j, dim in enumerate(layer_d):
        mixhop_layer.append(
          tf.keras.layers.Dense(dim, use_bias=use_bias, name='MixHop_L%i_pow%i' % (i, j)) )
      self.all_layers += mixhop_layer

  @property
  def trainable_variables(self):
    variables = []
    for layer in self.all_layers:
      variables += layer.trainable_variables
    return variables

  def __call__(self, x, adj, training=True, renorm=tf_utils.kipf_renorm_tf):
    if renorm:
      adj = renorm(adj)  # Add ones along diagonal & symmetric degree norm.
    t_adj = adj
    for k, mixhop_layer in enumerate(self.mixhop_layers):
      adj_pow_times_x = [x]
      for j in range(self.max_power):
        next_val = tf_utils.adj_times_x(t_adj, adj_pow_times_x[-1])
        adj_pow_times_x.append(next_val)

      layer_output = [sub_layer(adj_pow_times_x[j]) for j, sub_layer in enumerate(mixhop_layer)]
      layer_output = tf.concat(layer_output, axis=1)
      x = layer_output
      if k < len(self.mixhop_layers) - 1:
        x = tf.nn.relu(x)
        if training: x = tf.keras.layers.Dropout(0.5)(x)
    
    return x
    


class MixHopWithPSumClassifier:

  def __init__(self, num_classes, unused_num_feats, layer_dims, use_bias=False, **kwargs):
    self.num_classes = num_classes
    self.mixhop = MixHopGCN(layer_dims, use_bias=False, **kwargs)
    self.psum_weights = None

  @property
  def trainable_variables(self):
    return self.mixhop.trainable_variables + [self.psum_weights]

  def __call__(self, x, adj, training=True, renorm=tf_utils.kipf_renorm_tf):
    x = self.mixhop(x, adj, training=training, renorm=renorm)
    assert x.shape[1] % self.num_classes == 0
    
    num_groups = x.shape[1] // self.num_classes
    if self.psum_weights is None:
      self.psum_weights = tf.Variable(tf.zeros([num_groups]), name='psum_weights')

    output_divided = []
    for j in range(num_groups):
      output_divided.append(x[:, j*self.num_classes : (j+1)*self.num_classes ])
    
    output_divided = tf.stack(output_divided, 2)
    output_final = output_divided * tf.nn.softmax(self.psum_weights)
    output_final = tf.reduce_sum(output_final, -1)

    return output_final



class MixHopWithFCClassifier:

  def __init__(self, num_classes, unused_num_feats, layer_dims, use_bias=False, **kwargs):
    self.num_classes = num_classes
    self.mixhop = MixHopGCN(layer_dims, use_bias=False, **kwargs)
    self.fc = tf.keras.layers.Dense(num_classes)

  @property
  def trainable_variables(self):
    return self.mixhop.trainable_variables + self.fc.trainable_variables

  def __call__(self, x, adj, training=True, renorm=tf_utils.kipf_renorm_tf):
    x = self.mixhop(x, adj, training=training, renorm=renorm)
    x = tf.nn.relu(x)
    if training:
      x = tf.keras.layers.Dropout(0.5)(x)
    return self.fc(x)

