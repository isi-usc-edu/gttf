from torch import nn
import torch

class SGC(nn.Module):

    def __init__(self, nclass, nfeat, k=2):
        super(SGC, self).__init__()
        #SGC params
        nfeat = nfeat
        nclass = nclass
        self.k = k

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x, adj):
        for i in range(self.k):
            x = torch.spmm(adj, x)
        return self.W(x)

