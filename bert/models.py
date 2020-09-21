import torch.nn as nn

from transformer.transformer import TransformerEncoderBlock, get_padding_mask
from .blocks import BERTInputBlock


class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, max_seq=512, n_layers=12, n_heads=12, dropout=0.1, pad_idx=0):
        super().__init__()
        self._info = {'vocab_size': vocab_size,
                      'hidden_dim': hidden_dim,
                      'ff_hidden_dim': hidden_dim * 4,
                      'max_seq': max_seq,
                      'n_layers': n_layers,
                      'n_atttention_heads': n_heads,
                      'dropout': dropout,
                      'padding_idx': pad_idx}

        self.input = BERTInputBlock(vocab_size, hidden_dim, max_seq, dropout, pad_idx)
        self.encoder = TransformerEncoderBlock(n_layers, hidden_dim, hidden_dim * 4, n_heads, dropout)

    def forward(self, x_input, x_segment):
        x_mask = get_padding_mask(x_input)

        x_emb = self.input(x_input, x_segment)
        output = self.encoder(x_emb, x_mask)
        return output
