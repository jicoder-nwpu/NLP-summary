from curses.ascii import EM
from tkinter import N
from tkinter.messagebox import NO
from lib import *
import Embedding
import PositionalEncoder
from DecoderLayer import *
import Norm


class Decoder(nn.Module):
    def __init__(self, N, vocab_size, emb_dim, heads):
        super().__init__()
        self.N = N
        self.embed = Embedding(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim)
        self.layers = get_clones(DecoderLayer(emb_dim, heads), N)
        self.norm = Norm(emb_dim)
    
    def forward(self, trg, e_output, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_output, src_mask, trg_mask)
        return self.norm(x)