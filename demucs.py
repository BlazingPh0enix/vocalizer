import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CrossDomainTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossDomainTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        