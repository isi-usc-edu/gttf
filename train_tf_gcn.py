"""Trains GCN on Planetoid dataset."""

import json
import sys

from absl import flags, app
import numpy as np
import tensorflow as tf
import tqdm

from training_loops import datasets
from framework import accumulation, compact_adj, traversals
import models_tf
from models_tf import kipf_gcn, mixhop


flags.DEFINE_string('gcn_dataset', 'ind.cora', '')
flags.DEFINE_float('lr', 1e-2, 'Learning rate')
flags.DEFINE_float('l2_reg', 1e-4, 'L2 Regularization')
flags.DEFINE_integer('run_id', 0, 'Run ID to generate multiple runs for averaging.')
flags.DEFINE_string('out_dir', 'gcn_time_analysis', 'Directory where CSV files will be written')
flags.DEFINE_integer('eval_every', 1, 'Eval will be run every this many steps.')
flags.DEFINE_integer('epochs', 200, 'Eval will be run every this many steps.')
flags.DEFINE_string('model', 'kipf_gcn.KipfGCN', '')
flags.DEFINE_string('model_kwargs', '{}', 'JSON dict will be passed to model as **kwargs')
flags.DEFINE_string('fanout', '5x5', 'list of integers separated with "x".')


FLAGS = flags.FLAGS

def main(_):
  adj, allx, ally, test_id = datasets.read_planetoid_dataset(dataset_name=FLAGS.gcn_dataset)
  opt = tf.keras.optimizers.Adam(FLAGS.lr)  #1e-3 is probably best #1e-4 gave 82%

  allx = tf.convert_to_tensor(allx.todense())
  num_nodes = allx.shape[0]
  num_classes = ally.shape[1]
  labeled_nodes = np.arange(20*num_classes, dtype='int32')  # Planetoid
  validate_idx = np.arange(20*num_classes, 20*num_classes+500, dtype='int32')

  c_adj = compact_adj.CompactAdjacency(adj)
  eval_adj_tf = tf.cast(np.stack(c_adj.adj.nonzero(), -1), dtype=tf.int64)
  eval_adj_tf = tf.sparse.SparseTensor(
      indices=eval_adj_tf, values=tf.ones([eval_adj_tf.shape[0]], dtype=tf.float32),
      dense_shape=[num_nodes, num_nodes])
  
  model_class = models_tf.__dict__[FLAGS.model.split('.')[0]].__dict__[FLAGS.model.split('.')[1]]
  model_kwargs = json.loads(FLAGS.model_kwargs)
  gcn = model_class(num_classes, allx.shape[1], **model_kwargs)
  least_validate_loss = (99999999, 0)   # (Validate loss, test accuracy)
  fanouts = [int(f) for f in FLAGS.fanout.split('x')]

  tt = tqdm.tqdm(range(FLAGS.epochs))
  for epoch in tt:
    forest = traversals.np_traverse(c_adj, labeled_nodes, fanouts=fanouts)
    dense_shape = (num_nodes, num_nodes)
    sampled_adj = accumulation.SampledAdjacency.from_walk_forest(forest, dense_shape)
    trimmed_adj = sampled_adj.tf_trimmed
    trimmed_x = sampled_adj.tf_trim_x(allx)

    with tf.GradientTape(persistent=True) as tape:
      output = gcn(trimmed_x, trimmed_adj)
      labeled_logits = sampled_adj.tf_untrim_gather(output, labeled_nodes)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          labels=ally[labeled_nodes], logits=labeled_logits))
      variables = gcn.trainable_variables
      loss += tf.add_n([tf.reduce_sum(v ** 2)*FLAGS.l2_reg for v in variables])  # if 'kernel' in v.name

    grads = tape.gradient(loss, variables)
    opt.apply_gradients(zip(grads, variables))

    if epoch % FLAGS.eval_every == 0:
      #timer.stop()
      output = gcn(allx, eval_adj_tf, training=False)
      vloss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              labels=ally[validate_idx],
              logits=tf.gather(output, validate_idx))).numpy()

      preds = tf.argmax(tf.gather(output, test_id), 1).numpy()
      test_accuracy = (preds == ally[test_id].argmax(1)).mean()
      least_validate_loss = min(least_validate_loss, (vloss, test_accuracy))
      tt.set_description('Test: %g (@ best validate=%g)' % (test_accuracy, least_validate_loss[1]))
      #first_batch_offset = first_batch_offset or timer.total
      #csv_out.append('%i,%f,%f' % (step, timer.total - first_batch_offset, least_validate_loss[1]))
      #timer.start()


if __name__ == '__main__':
  app.run(main)