import numpy as np
import torch

from torch import nn

from utils.common import asym_dot


class DeepWalk(nn.Module):

    def __init__(self, num_nodes, emb_dim, window_size):
        super(DeepWalk, self).__init__()

        self.num_nodes = num_nodes
        self.window_size = window_size

        # self.Z = torch.from_numpy(np.random.uniform(low=-0.01, high=0.01, size=[num_nodes, emb_dim]))
        self.Z = nn.Embedding(self.num_nodes, emb_dim)
        self.Z.weight = nn.Parameter(torch.from_numpy(np.random.uniform(low=-0.01, high=0.01, size=[num_nodes, emb_dim])), requires_grad=True)

    def embeddings(self):
        return (self.Z,)

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

        center_embeds = self.Z(forest[0][:, 0])

        ## POSITIVES.
        mean_context_embeds = None
        total_weight = 0

        for w, context_nodes in enumerate(forest[1:]):
            context_embeds = self.Z(context_nodes)
            if mean_context_embeds is None:
                mean_context_embeds = context_embeds[:, 0, :] * (self.window_size - w) / self.window_size
            else:
                mean_context_embeds += context_embeds[:, 0, :] * (self.window_size - w) / self.window_size
            total_weight += (self.window_size - w) / self.window_size


        # ## NEGATIVES.
        negs = torch.from_numpy(np.random.choice(self.num_nodes, size=(len(forest[0]), negative_samples)))
        neg_embeddings = self.Z(negs)

        # # shape: num pos X num neg
        # # center_dot_negs = tf.reduce_sum(tf.expand_dims(center_embeds, 1) * neg_embeddings, axis=-1)
        center_dot_negs = asym_dot(
            center_embeds[:, :, None],
            torch.transpose(neg_embeddings, 2, 1).detach(),
            sum_fn=torch.sum,
            dim=1
        )

        center_dot_context = asym_dot(center_embeds, mean_context_embeds, sum_fn=torch.sum, dim=1)
        neg_loss = torch.logsumexp(center_dot_negs, dim=1)
        pos_loss = -center_dot_context
        total_loss = torch.mean(pos_coeff * pos_loss + neg_loss)

        return total_loss

    def forward(self, x):
        return self.Z(x)

