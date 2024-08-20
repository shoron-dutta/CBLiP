import torch.nn as nn
import torch, copy
from math import sqrt
import torch.nn.functional as F

class Norm(nn.Module):
    def __init__(self, effective_d, eps = 1e-6):
        super().__init__()
    
        self.size = effective_d
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ffn, dropout = 0.1):
        super().__init__() 
        
        self.linear_1 = nn.Linear(d_model, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_ffn, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def attention(q, k, v, d_k,  bias_key, bias_value, mask=None, dropout=None):
    
    # connection_bias.shape [b, seq_len, seq_len, d]
    # q, k, v [b, nh, seq_len, d_k]
    b, nh, s, _ = q.shape
    
    # custom connection biased attention score computation
    # [b, nh, seq_len, 1, d] x [b, nh, seq_len, d, seq_len] -> [b, nh, seq_len,1,seq_len]
    
    connection_biased_k = k.unsqueeze(dim=2) + bias_key.reshape(b,s,s,nh,d_k).transpose(1,3) # connection_biased_k: [b, nh, seq_len, seq_len, d]
    scores = torch.matmul(q.unsqueeze(dim=3), connection_biased_k.transpose(-2, -1)).squeeze() /  sqrt(d_k) # [b, nh, seq_len, seq_len]
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(-1)
    
    scores = scores.masked_fill_(mask == 1, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)

    connection_biased_value = v.unsqueeze(dim=2) + bias_value.reshape(b,s,s,nh,d_k).transpose(1,3)
    output = torch.matmul(scores.unsqueeze(dim=3), connection_biased_value)
    return output.squeeze()


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, connection_bias_k=None, connection_bias_v=None, mask=None):
        
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # k = (W_k)(x)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # q = (W_q)(x)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) # v = (W_v)(x)
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = attention(q, k, v, self.d_k,  connection_bias_k, connection_bias_v, mask, self.dropout) # [b, nh, seq_len, d]
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output
class EncoderLayer(nn.Module):
    def __init__(self, d_model, dim_ffn, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, dim_ffn)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, connection_bias_k=None, connection_bias_v=None, mask=None):
        x2 = self.norm_1(x)
        # print('inside Encoder Layer:')
        # print(x.shape, self.dropout_1(self.attn(x2,x2,x2, connection_bias_k, connection_bias_v, mask)).shape)
        x = x + self.dropout_1(self.attn(x2,x2,x2,  connection_bias_k, connection_bias_v, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, dim_ffn, nl, heads):
        super().__init__()
        self.nl = nl
        self.layers = get_clones(EncoderLayer(d_model, dim_ffn, heads), nl)
        self.norm = Norm(d_model)
    def forward(self, x, connection_bias_k=None, connection_bias_v=None, mask=None):
        
        for i in range(self.nl):
            x = self.layers[i](x, connection_bias_k, connection_bias_v, mask)
        return self.norm(x)
