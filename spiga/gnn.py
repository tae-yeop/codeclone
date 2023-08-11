from copy import deepcopy

import torch
from torch import nn

def MLP(channels : list):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i-1], channels[i], kernel_size=1, bias=True))

        if i < (n-1):
            layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)



print(MLP([3,4,5,6,7,8]))
# Sequential(
#   (0): Conv1d(3, 4, kernel_size=(1,), stride=(1,))
#   (1): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): Conv1d(4, 5, kernel_size=(1,), stride=(1,))
#   (4): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU()
#   (6): Conv1d(5, 6, kernel_size=(1,), stride=(1,))
#   (7): BatchNorm1d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (8): ReLU()
#   (9): Conv1d(6, 7, kernel_size=(1,), stride=(1,))
#   (10): BatchNorm1d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (11): ReLU()
#   (12): Conv1d(7, 8, kernel_size=(1,), stride=(1,))
# )

class Attention(nn.Module):
    def __init__(self, num_heads, feature_dim):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.dim = feature_dim // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        # Q, K, V
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, qeury, key, value):
        # q : [B, feature_dim,  seq_len] : kernel =1 짜리 conv로 convolving
        batch_dim = qeury.size(0)
        # projection
        query, key, value = [l(x).view(batch_dim, self.dim, ) for l, x in zip(self.proj, (query, key, value))]
        x, prob = self.attention(query, key, value)
        return self.merge(x.contiguos().view(batch_dim, self.dim*self.dim * self.num_heads, -1)), prob

    def attention(self, query, key, value):
        dim = query.shape[1]
        # batched matrix multiplication
        scores = torch.einsum('bdhn, bdhm -> bhnm', query, key) / dim ** .5
        prob = F.softmax(scores, dim=-1)
        return torch.einsum('bhnm, bdhn->bdhn', prob, value), prob
    
class MessagePassing(nn.Module):
    def __init__(self, feature_dim, num_heads, out_dim=None):
        super().__init__()
        self.attn = Attention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, out_dim])

    def forward(self, features):
        message, prob = self.attn()

class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()

        num_heads_in = num_heads
        self.reshpae = None
        if input_dim != output_dim:
            for num_heads_in in range(num_heads, 0 , -1):
                if input_dim % num_heads_in == 0:
                    break
            self.reshape = MLP([input_dim, output_dim])

        