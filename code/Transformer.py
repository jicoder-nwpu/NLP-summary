from lib import *
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, emb_dim, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_dim, N, heads)
        self.decoder = Decoder(N, trg_vocab, emb_dim, heads)
        self.out = nn.Linear(emb_dim, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_output = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_output, src_mask, trg_mask)
        output = self.out(d_output)
        return output