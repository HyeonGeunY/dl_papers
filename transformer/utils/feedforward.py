import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, activation=nn.ReLU(), is_gated: bool = False, bias1: bool = True, bias2: bool = True, bias_gate: bool = True):
        """
        d_model: token 임배딩의 차원 수
        d_ff: FFN의 hidden layer 차원 수
        dropout: hidden laeyr의 dropout probability
        is_gated: hidden layer의 gated 여부
        bias1: 첫 번째 hidden layer가 bias를 갖는 지 여부
        bias2: 두 번째 hidden layer가 bias를 갖는 지 여부
        bias_gate: fully connected layer의 gate가 학습 가능하나 bias를 갖는지 여부
        """

        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)

        self.dropout = nn.Dropout(dropout)

        self.activation = activation

        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)
        
    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))

        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        
        x = self.dropout(x)

        return self.layer2(x)