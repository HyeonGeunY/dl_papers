import torch
import numpy as np
from torch import nn
from typing import List, Optional


class MultiHeadAttentionUnit(nn.Module):
    """
    multi-head-attention에서 key, query, value 위한 unit 
    """
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        """
        d_model: input 임배딩 차원
        heads: head 개수
        d_k: 각 head의 차원
        논문에서는 d_model == heads * d_k
        """    
        super().__init__()
            
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k
        
    
    def forward(self, x: torch.Tensor):
        """
        x: input vector (seq_len, batch_size, d_model) or (batch_size, d_model)
        
        output_size: (seq_len, batch_size, heads, d_k) or (batch_size, heads, d_k)
        """
        head_shape = x.shape[:-1] # 연산 후 각 헤드 별로 분할하기 위함.
        
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        
        self.d_k = d_model // heads
        self.heads = heads
        
        self.query = MultiHeadAttentionUnit(d_model, heads, self.d_k, bias=bias)
        self.value = MultiHeadAttentionUnit(d_model, heads, self.d_k, bias=bias)
        self.key = MultiHeadAttentionUnit(d_model, heads, self.d_k, bias=bias)
        
        self.softmax = nn.Softmax(dim=1)  
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / np.sqrt(self.d_k)
        
        self.attn = None # logging을 위한 attention 저장
    
    
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        i: ith seq
        j: jth seq
        b: batch_size
        h: n_heads
        """
        
        return torch.einsum('ibhd, jbhd -> ijbh', query, key) 
    
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        mask: (seq_len_q, seq_len_k, batch_size)
        query, key: (seq_len, batch_size, n_heads, d_model)
        """
        
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        
        mask = mask.unsqueeze(-1) # 모든 헤드에 대해 동일한 마스크 확장
        
        return mask 
        
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: 
                Optional[torch.Tensor]=None):
        """
        
        """
        seq_len, batch_size, _ = query.shape # (seq_len_q, batch_size, d_model)
        
        
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
            
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        scores = self.get_scores(query, key)
        scores *= self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
            
        attn = self.softmax(scores)
        self.attn = attn.detach().clone()
        attn = self.dropout()
        
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)
        
        x = x.reshape(seq_len, batch_size, -1) # n_heads * n_dim => d_model
        
        return self.ouput(x)