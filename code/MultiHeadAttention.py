#多头注意力机制实现  q k ——> scores mask v -> output
from lib import *


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads, dropout = 0.1):
        super().__init__()

        self.emb_dim = emb_dim
        self.d_k = self.emb_dim // heads    #每个头向量维度， 用词向量维度作为线性层输出维度
        self.h = heads

        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, emb_dim)

    def forward(self, q, k, v, mask = None):

        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        k = self.q_linear(k).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = attention(q, k, v, self.d_k, mask, self.dropout)
        conact = output.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)    #contiguous 返回连续内存tensor
        output = self.out(conact)
        return output


#求注意力机制和
def attention(q, k, v, d_k, mask = None, dropout = None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)     #matmul 存在广播机制

    if mask is not None:
        mask = mask.unsqueeze(1)      #？
        scores = scores.masked_fill(mask == 0, -1e9)     #mask为boolean tensor，mask为true的地方填入 value
    
    scores = F.softmax(scores, dim = -1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


def main():
    mul_h_a = MultiHeadAttention(4, 2)
    input = torch.randn(1, 7, 4)
    print(input)
    print(mul_h_a(input, input, input))


if __name__ == '__main__':
    main()