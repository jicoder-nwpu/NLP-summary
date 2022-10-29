# torch.nn Embedding类 提供可训练的词向量库
# import torch
# import torch.nn as nn
from lib import *


# embedding = nn.Embedding(10, 3)

# print(embedding.weight)

# input = torch.LongTensor([1])
# print(embedding(input))
class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_dim) :
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
    def forward(self, x):
        return self.embed(x)

def main():
    embedding = Embedder(10, 5)
    print(embedding.embed.weight)
    input = torch.tensor([8,6,2,4])
    print(embedding(input))

if __name__ == '__main__':
    main()