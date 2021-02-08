import tensorflow as tf

from utils import tf_utils


class KipfGCN:

  def __init__(self, num_classes, unused_num_feats, hidden_dim=64, **kwargs):
    self.num_classes = num_classes
    self.layer1 = tf.keras.layers.Dense(hidden_dim, use_bias=False)
    self.dropout1 = tf.keras.layers.Dropout(0.5)
    self.layer2 = tf.keras.layers.Dense(self.num_classes, use_bias=False)
  
  @property
  def trainable_variables(self):
    return self.layer1.trainable_variables + self.layer2.trainable_variables

  def __call__(self, x, adj, training=True, renorm=tf_utils.kipf_renorm_tf):
    if renorm:
      adj = renorm(adj)  # Add ones along diagonal & symmetric degree norm.
    assert isinstance(adj, tf.sparse.SparseTensor)
    adj = adj
    
      
    
    x = self.layer1(tf_utils.adj_times_x(adj, x))  # GC layer
    x = tf.nn.relu(x)
    if training: x = tf.keras.layers.Dropout(0.5)(x)
    return self.layer2(tf_utils.adj_times_x(adj, x))  # GC layer



class SimpleGCN:

  def __init__(self, num_classes, unused_num_feats, **kwargs):
    self.num_classes = num_classes
    self.dropout1 = tf.keras.layers.Dropout(0.5)
    self.layer2 = tf.keras.layers.Dense(self.num_classes, use_bias=False)
  
  @property
  def trainable_variables(self):
    return self.layer2.trainable_variables

  def __call__(self, x, adj, training=True, renorm=tf_utils.kipf_renorm_tf):
    if renorm:
      adj = renorm(adj)  # Add ones along diagonal & symmetric degree norm.
    assert isinstance(adj, tf.sparse.SparseTensor)
    adj = adj
    
    x = tf_utils.adj_times_x(adj, x)
    #if training: x = tf.keras.layers.Dropout(0.8)(x)
    return self.layer2(tf_utils.adj_times_x(adj, x))  # GC layer
