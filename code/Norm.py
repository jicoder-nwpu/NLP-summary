# 归一化层 强制回归正态分布
from lib import *


class Norm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-6):
        super().__init__()
        self.size = emb_dim
        self.alpha = nn.Parameter(torch.ones(self.size))    #Parameter 将tensor转化为模型参数模式，可以参与模型训练，此处可以认为手动线性层
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    ''' 
    mean 输出均值 keepdim为True保持原维度,dim为1,否则被压缩
    std 计算标准差
    归一化计算公式为  (x - 均值) / (标准差 + eps)  再过线性层
    '''
    def forward(self, x):
        # [4] * [7, 4] -> [7, 4]
        norm = self.alpha * (x - x.mean(dim = -1, keepdim=True)) \
        / (x.std(dim = -1, keepdim=True) + self.eps) + self.bias
        return norm


def main():
    norm = Norm(4)
    input = torch.randn((7, 4))
    print(input)
    print(norm(input))


if __name__ == '__main__':
    main()