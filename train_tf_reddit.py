"""Trains GCN on Planetoid dataset."""

import copy
import json
import os
import sys

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from absl import flags, app
import numpy as np
import tensorflow as tf
import tqdm

from framework import accumulation, compact_adj, traversals
import timing
import models_tf
from models_tf import kipf_gcn, mixhop

from training_loops import datasets


flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('l2_reg', 1e-5, 'L2 Regularization')
flags.DEFINE_string('out_dir', '~/data/gttf/output/reddit', 'Directory where CSV files will be written')
flags.DEFINE_integer('eval_every', 1, 'Eval will be run every this many steps.')
flags.DEFINE_integer('epochs', 200, '')

flags.DEFINE_integer('batch_size', 512, '')

flags.DEFINE_string('model', 'kipf_gcn.KipfGCN', '')
flags.DEFINE_string('model_kwargs', '{}', 'JSON dict will be passed to model as **kwargs.')
flags.DEFINE_string('fanout', '5x5', 'list of integers separated with "x".')
flags.DEFINE_string('run_id', '0', 'String to identify this run.')


FLAGS = flags.FLAGS

def output_filename():
  out_dir = os.path.expanduser(FLAGS.out_dir)
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  out_filename = os.path.join(out_dir, 'reddit-%s-%s-%s.csv' % (FLAGS.model, FLAGS.fanout, FLAGS.run_id))
  return out_filename



def main(_):
  opt = tf.keras.optimizers.Adam(FLAGS.lr)  #1e-3 is probably best #1e-4 gave 82%
  directory = os.path.expanduser('~/data/graphs/reddit')
  
  allx = np.load(os.path.join(directory, 'feat_data.npy'))
  allx -= allx.mean(0, keepdims=True)
  allx /= allx.std(0, keepdims=True)
  allx = np.array(allx, 'float32')
  allx = tf.convert_to_tensor(allx)
  ally = np.load(os.path.join(directory, 'labels.npy'))
  ally = ally[:, 0]
  
  train_ids = np.load(os.path.join(directory, 'train_ids.npy'))
  val_ids = np.load(os.path.join(directory, 'val_ids.npy'))
  test_ids = np.load(os.path.join(directory, 'test_ids.npy'))

  trainy = copy.deepcopy(ally)
  trainy[val_ids] = -1
  trainy[test_ids] = -1


  train_id_set = set(train_ids)
  train_id_set.update(val_ids)
  test_id_set = set(test_ids)

  num_classes = ally.max() + 1

  c_adj = compact_adj.CompactAdjacency.from_file(os.path.join(directory, 'cadj.npz'))
  num_nodes = c_adj.adj.shape[0]
  eval_adj_tf = tf.cast(np.stack(c_adj.adj.nonzero(), -1), dtype=tf.int64)
  eval_adj_tf = tf.sparse.SparseTensor(
      indices=eval_adj_tf, values=tf.ones([eval_adj_tf.shape[0]], dtype=tf.float32),
      dense_shape=[num_nodes, num_nodes])



  model_class = models_tf.__dict__[FLAGS.model.split('.')[0]].__dict__[FLAGS.model.split('.')[1]]
  model_kwargs = json.loads(FLAGS.model_kwargs)
  gcn = model_class(num_classes, allx.shape[1], **model_kwargs)
  least_validate_loss = (99999999, 0)   # (Validate loss, test accuracy)
  fanouts = [int(f) for f in FLAGS.fanout.split('x')]

  first_batch_offset = None
  timer = timing.WallClockTimer()
  timer.start()  # Will be stopped only when evaluating test accuracy.
  csv_out = ['epoch,time,accuracy']
  for epoch in range(FLAGS.epochs):
    permutation = np.random.permutation(train_ids)
    tt = tqdm.tqdm(range(0, len(permutation), FLAGS.batch_size))
    for starti in tt:
      endi = starti + FLAGS.batch_size
      batch = permutation[starti:endi]
      forest = traversals.np_traverse(c_adj, batch, fanouts=fanouts)
      dense_shape = (num_nodes, num_nodes)
      sampled_adj = accumulation.SampledAdjacency.from_walk_forest(forest, dense_shape)
      trimmed_adj = sampled_adj.tf_trimmed
      trimmed_x = sampled_adj.tf_trim_x(allx)

      with tf.GradientTape(persistent=True) as tape:
        output = gcn(trimmed_x, trimmed_adj)
        labeled_logits = sampled_adj.tf_untrim_gather(output, batch)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(ally[batch], num_classes), logits=labeled_logits))
        variables = gcn.trainable_variables
        loss += tf.add_n([tf.reduce_sum(v ** 2)*FLAGS.l2_reg for v in variables])  # if 'kernel' in v.name

      grads = tape.gradient(loss, variables)
      opt.apply_gradients(zip(grads, variables))

    if (epoch+1) % FLAGS.eval_every == 0:
      timer.stop()
      with tf.device('cpu:0'):
        output = gcn(allx, eval_adj_tf, training=False)
        vloss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(ally[val_ids], num_classes),
                logits=tf.gather(output, val_ids))).numpy()

        preds = tf.argmax(tf.gather(output, test_ids), 1).numpy()
        test_accuracy = (preds == ally[test_ids]).mean()
        least_validate_loss = min(least_validate_loss, (vloss, test_accuracy))
      print('%s-%s-%s] Test: %g (@ best validate=%g)' % (
          FLAGS.model, FLAGS.fanout, FLAGS.run_id, test_accuracy, least_validate_loss[1]))
      first_batch_offset = first_batch_offset or timer.total
      csv_out.append('%i,%f,%f' % (epoch, timer.total - first_batch_offset, least_validate_loss[1]))
      timer.start()

  with open(output_filename(), 'w') as fout:
    fout.write('\n'.join(csv_out))
    print('wrote ' + output_filename())

if __name__ == '__main__':
  app.run(main)