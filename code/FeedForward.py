#前馈层
from lib import *


class FeedForward(nn.Module):
    def __init__(self, emb_dim, d_ff = 2048, dropout = 0.1):
        super().__init__()

        self.linear_1 = nn.Linear(emb_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, emb_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))     #负数为0 非负数不变
        x = self.linear_2(x)
        return x
    

def main():
    feed = FeedForward(4, 16)      #torch Module 通过.cuda()方法将模型参数放入GPU
    input = torch.randn(1, 7, 4)
    print(input)
    print(feed(input))


if __name__ == '__main__':
    main()