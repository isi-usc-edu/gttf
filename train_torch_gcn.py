"""Trains GCN on Planetoid dataset."""

import json
import sys

from absl import flags, app
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from training_loops import datasets
from framework import accumulation, compact_adj, traversals
import models_pytorch
from models_pytorch import gcnii, simple_gcn
from utils import torch_utils


flags.DEFINE_string('gcn_dataset', 'ind.cora', '')
flags.DEFINE_float('lr', 1e-2, 'Learning rate')
flags.DEFINE_float('l2_reg', 1e-4, 'L2 Regularization')
flags.DEFINE_integer('run_id', 0, 'Run ID to generate multiple runs for averaging.')
flags.DEFINE_string('out_dir', 'gcn_time_analysis', 'Directory where CSV files will be written')
flags.DEFINE_integer('eval_every', 1, 'Eval will be run every this many steps.')
flags.DEFINE_integer('epochs', 200, 'Total number of dataset iterations.')
flags.DEFINE_string('model', 'gcnii.GCNII', '')
flags.DEFINE_string('model_kwargs', '{}', 'JSON dict will be passed to model as **kwargs')
flags.DEFINE_string('fanout', '5x5', 'list of integers separated with "x".')
flags.DEFINE_string('device', 'cpu', 'Should be "cpu" or "cuda" or any valid torch device name')

FLAGS = flags.FLAGS



def main(_):
  if FLAGS.device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA not available, using CPU.")
    FLAGS.device = "cpu"
  adj, allx, ally, test_id = datasets.read_planetoid_dataset(dataset_name=FLAGS.gcn_dataset)
  allx = torch.from_numpy(allx.todense()).to(FLAGS.device)
  num_nodes = allx.shape[0]
  num_classes = ally.shape[1]
  labeled_nodes = np.arange(20*num_classes, dtype='int32')  # Planetoid
  validate_idx = np.arange(20*num_classes, 20*num_classes+500, dtype='int32')
  labels = torch.max(torch.LongTensor(ally).to(FLAGS.device), dim=1)[1] 

  c_adj = compact_adj.CompactAdjacency(adj)
  eval_adj = torch_utils.kipf_renorm_sp(adj)
  eval_adj_torch = torch_utils.sparse_mx_to_torch_sparse_tensor(eval_adj).to(FLAGS.device)

  model_class = models_pytorch.__dict__[FLAGS.model.split('.')[0]].__dict__[FLAGS.model.split('.')[1]]
  model_kwargs = json.loads(FLAGS.model_kwargs)
  gcn = model_class(num_classes, allx.shape[1], **model_kwargs)
  gcn = gcn.to(FLAGS.device)
  if FLAGS.l2_reg > 0:
    optimizer = optim.Adam(gcn.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.l2_reg)
  else:
    optimizer = optim.Adam(gcn.parameters(), lr=FLAGS.lr)
  

  least_validate_loss = (99999999, 0)   # (Validate loss, test accuracy)
  fanouts = [int(f) for f in FLAGS.fanout.split('x')]

  tt = tqdm.tqdm(range(FLAGS.epochs))
  for epoch in tt:
    forest = traversals.np_traverse(c_adj, labeled_nodes, fanouts=fanouts)
    dense_shape = (num_nodes, num_nodes)
    sampled_adj = accumulation.SampledAdjacency.from_walk_forest(forest, dense_shape)
    trimmed_adj = sampled_adj.csr_trimmed
    trimmed_adj = torch_utils.kipf_renorm_sp(trimmed_adj)
    trimmed_adj = torch_utils.sparse_mx_to_torch_sparse_tensor(trimmed_adj)
    trimmed_adj = trimmed_adj.to(FLAGS.device)
    trimmed_x = sampled_adj.torch_trim_x(allx)
    trimmed_x = trimmed_x.to(FLAGS.device)

    # Optimization step
    gcn.train()
    optimizer.zero_grad()
    gcn_output = gcn(trimmed_x, trimmed_adj)

    if FLAGS.model.split('.')[1] == 'SGC':  # THIS IS A HACK. SHOULD BE REFACTORED
      # Isn't this equivalent to the else-clause?
      loss_train = F.cross_entropy(sampled_adj.torch_untrim_gather(gcn_output, labeled_nodes), labels[labeled_nodes])
    else:
      logits = F.log_softmax(gcn_output, dim=1)
      loss_train = F.nll_loss(sampled_adj.torch_untrim_gather(logits, labeled_nodes), labels[labeled_nodes])
    loss_train.backward()
    optimizer.step()

    if epoch % FLAGS.eval_every == 0:
      #timer.stop()
      gcn.eval()
      output = gcn(allx, eval_adj_torch)
      vloss = F.nll_loss(output[validate_idx], labels[validate_idx]).detach().cpu().numpy()

      preds = torch.max(output[test_id], 1)[1].detach().cpu().numpy()
      test_accuracy = (preds == ally[test_id].argmax(1)).mean()
      least_validate_loss = min(least_validate_loss, (vloss, test_accuracy))
      tt.set_description('Test: %g (@ best validate=%g)' % (test_accuracy, least_validate_loss[1]))
      #first_batch_offset = first_batch_offset or timer.total
      #csv_out.append('%i,%f,%f' % (step, timer.total - first_batch_offset, least_validate_loss[1]))
      #timer.start()


if __name__ == '__main__':
  app.run(main)