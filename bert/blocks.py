import torch
import torch.nn as nn


class BERTInputBlock(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, max_seq=512, dropout=0.1, pad_idx=0, position_padding=True):
        """
        BERT Input Block
        The reason why segment embedding is initialized with 3 vocab size is because of the pad_idx
        :param int vocab_size: corpus vocab size
        :param hidden_dim: d_model in Transformer
        :param max_seq: max sequence length
        :param dropout: dropout probability
        :param pad_idx: padding idx, default 0
        """
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, pad_idx)
        self.positional_emb = nn.Embedding(max_seq + 1, hidden_dim, pad_idx)
        self.segment_emb = nn.Embedding(3, hidden_dim, pad_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.position_padding = position_padding

    def forward(self, x, x_seg, pos_pad=True):
        pos = torch.arange(start=1, end=x.shape[1] + 1, dtype=torch.long, device=x.device, requires_grad=False)
        pos = pos.unsqueeze(0).expand_as(x)  # (T,) -> (N, T)
        if self.position_padding:
            pos = pos.masked_fill(x == 0, value=0)

        x = self.token_emb(x) + self.positional_emb(pos)
        if x_seg is not None:
            x += self.segment_emb(x_seg)
        return self.dropout(x)
