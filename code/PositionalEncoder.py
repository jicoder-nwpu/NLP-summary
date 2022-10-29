# import math
# import torch
# import torch.nn as nn
from lib import *

class PositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_seq_len = 80, drop_out = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        pe = torch.zeros(max_seq_len, emb_dim)
        self.dropout = nn.Dropout(drop_out)
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):   # emb_dim 在这种情况下，仅仅接受偶数
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / emb_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / emb_dim)))
        pe = pe.unsqueeze(0) #  torch.Size([80, 4])  ->   torch.Size([1, 80, 4])\
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.emb_dim)
        seq_len = x.size(1)
        x = x + torch.tensor(self.pe[:, : seq_len], requires_grad=False).cuda()  #tensor相加
        return self.dropout(x)

def main():
    pos_encoder = PositionalEncoder(4)
    input = torch.randn([5, 7, 4]).cuda()
    print(pos_encoder(input))


if __name__ == '__main__':
    main()