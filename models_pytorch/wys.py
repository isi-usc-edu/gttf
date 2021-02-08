import numpy as np
import torch

from torch import nn
import torch.nn.functional as F

from utils.common import asym_dot


class WYS(nn.Module):

    def __init__(self, num_nodes, emb_dim, window_size):
        super(WYS, self).__init__()

        self.num_nodes = num_nodes
        self.window_size = window_size

        self.Q = torch.nn.Parameter(torch.zeros(window_size,))
        self.L = nn.Embedding(self.num_nodes, emb_dim//2)
        self.L.weight = nn.Parameter(torch.from_numpy(np.random.uniform(low=-0.01, high=0.01, size=[num_nodes, emb_dim])), requires_grad=True)
        self.R = nn.Embedding(self.num_nodes, emb_dim//2)
        self.R.weight = nn.Parameter(torch.from_numpy(np.random.uniform(low=-0.01, high=0.01, size=[num_nodes, emb_dim])), requires_grad=True)

    def embeddings(self):
        return self.L, self.R

    def loss(self, forest, negative_samples=5, pos_coeff=5):
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

        l0 = self.L(forest[0][:, 0])
        r0 = self.R(forest[0][:, 0])
        Q_softmax = F.softmax(self.Q, dim=0)

        ## POSITIVES.
        obj_terms = []
        for w, context_nodes in enumerate(forest[1:]):
            l = self.L(context_nodes)
            r = self.R(context_nodes)
            pos_term = -(Q_softmax[w])*torch.mean(F.logsigmoid(0.5*torch.sum(l0.unsqueeze(1) * r, axis=2) + 0.5*torch.sum(r0.unsqueeze(1) * l, axis=2)))
            obj_terms.append(pos_term)
            
        pos_obj_term = torch.sum(torch.stack(obj_terms))
          

        # ## NEGATIVES.
        negs = torch.from_numpy(np.random.choice(self.num_nodes, size=(len(forest[0]), negative_samples)))
        neg_l = self.L(negs)
        neg_r = self.R(negs)
        negp = 0.5*torch.sum(l0.unsqueeze(1) * neg_r, axis=2) + 0.5*torch.sum(r0.unsqueeze(1) * neg_l, axis=2)
        neg_obj_mean = torch.mean(-F.logsigmoid( -negp))

        # ## REGULARIZER
        reg_obj = 0.5*torch.sum(self.Q**2)
        reg_obj += 1e-5 * torch.sum(self.L.weight**2)
        reg_obj += 1e-5 * torch.sum(self.R.weight**2)

        total_loss = pos_obj_term + neg_obj_mean + reg_obj

        return total_loss

    def forward(self, x):
        return self.L(x), self.R(x)

