# decoder layer
from ctypes.wintypes import HACCEL
from statistics import mode
from lib import *
import Norm
import MultiHeadAttention
import FeedForward
import copy


class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)
        self.norm_3 = Norm(emb_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(emb_dim, heads, dropout)
        self.attn_2 = MultiHeadAttention(emb_dim, heads, dropout)
        self.ff = FeedForward(emb_dim)

    def forward(self, x, e_outputs, src_mask, target_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, target_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

# 堆叠多层输入、输出层
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])