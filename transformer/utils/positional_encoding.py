# written by https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/940b3c01fc87c9487ad9478eec09f0167e177e2a/labml_nn/transformers/positional_encoding.py#L1

import math

import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer('positinonal_encodings', get_positional_encoding(d_model, max_len), False) # state_dict()에 포함시키지 않음.

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x
        
        
def get_positional_encoding(d_model: int, max_len: int = 5000):
    
    encodings = torch.zeros(max_len, d_model) # 빈 인코딩 벡터
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # positional index (max_len, 1)

    two_i = torch.arange(0, d_model, 2, dtype=torch.float32) # (d_model)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term) # (max_len, d_model)
    encodings[:, 1::2] = torch.cos(position * div_term)

    # batch 차원 추가: (S, E) -> (S, B, E)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    
    return encodings
    
    
def _test_positional_encoding():
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    pe = get_positional_encoding(20, 100)
    plt.plot(np.arange(100), pe[:, 0, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.title("Positional encoding")
    plt.show()
    

if __name__ == "__main__":
    _test_positional_encoding()