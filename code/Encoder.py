import EncodeLayer
import Embedding
import PositionalEncoder
import Norm
from DecoderLayer import get_clones


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, N, heads):
        super().__init__()

        self.N = N
        self.embed = Embedding(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim)
        self.layers = get_clones(EncodeLayer(emb_dim, heads), N)
        self.norm = Norm(emb_dim)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)