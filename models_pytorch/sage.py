import torch
import torch.nn as nn


class GraphSage(nn.Module):
	def __init__(self, feat_data, num_classes, hidden_dim1, hidden_dim2, dropout):
		super(GraphSage, self).__init__()

		self.num_classes = num_classes
		self.feat_dim = feat_data.shape[1]		
		self.hidden_dim1 = hidden_dim1
		self.hidden_dim2 = hidden_dim2
		
		
		self.layer1 = nn.Linear(2 * feat_data.shape[1], hidden_dim1, bias=False)
		self.layer2 = nn.Linear(2 * hidden_dim1, hidden_dim2, bias=False)
		self.output_layer = nn.Linear(hidden_dim2, num_classes, bias=False)
		
		self.embedding_table = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
		self.embedding_table.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
		
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=dropout)


	def forward(self, T):
		feat_0 = self.embedding_table(T[0]) # Of shape torch.Size([|B|, feat_dim])
		feat_1 = self.embedding_table(T[1]) # Of shape torch.size(|B|, fanouts[0], feat_dim)

		# Depth 1
		x = self.embedding_table(T[1]).mean(dim=1) # Of shape torch.size(|B|, feat_dim)
		feat_0 = torch.cat((feat_0, x), dim=1) # Of shape torch.size(|B|, 2 * feat_dim)
		feat_0 = self.relu(self.layer1(feat_0)) # Of shape torch.size(|B|, hidden_dim1)
		feat_0 = self.dropout(feat_0)

		# Depth 2
		x = self.embedding_table(T[2]).mean(dim=1) # Of shape torch.size(|B|*fanouts[0], feat_dim)	
		feat_1 = torch.cat((feat_1.reshape(-1, self.feat_dim) , x), dim=1) # Of shape torch.size(|B|*fanouts[0], 2 * feat_dim)
		feat_1 = self.relu(self.layer1(feat_1)) # Of shape torch.size(|B|*fanouts[0], hidden_dim1)
		feat_1 = self.dropout(feat_1)

		# Combine
		feat_1 = feat_1.reshape(T[0].shape[0], -1, self.hidden_dim1).mean(dim=1) # Of shape torch.size([|B|, hidden_dim_1])
		combined = torch.cat((feat_0, feat_1), dim=1) # Of shape torch.Size(|B|, 2 * hidden_dim1)
		embedding = self.relu(self.layer2(combined)) # Of shape torch.Size(|B|, hidden_dim2)


		# Output class scores
		scores = self.output_layer(embedding)

		return scores
