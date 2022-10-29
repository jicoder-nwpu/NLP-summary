# encoder one layer
from lib import *
import MultiHeadAttention
import FeedForward
import Norm


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, heads, dropout = 0.1):
        super().__init__()

        #每一层的输入都需要归一化，每一层的输出都dropout
        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, heads, dropout)
        self.ff = FeedForward(emb_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    # 加入残差 避免梯度消失
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x