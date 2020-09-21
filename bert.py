from transformer.transformer import *



class BERTModel:
    def __init__(self, vocab_size, hidden_dim=768, max_seq_len=512, n_layers=12, n_heads=12, dropout=0.1, pad_idx=0):
        self.input = BERTInputBlock(vocab_size)
        self.encoders = nn.ModuleList(
            [TransformerEncoderBlock(n_layers, hidden_dim, hidden_dim * 4, n_heads, dropout) for _ in range(n_layers)])