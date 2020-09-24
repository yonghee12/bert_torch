from torchsummary import summary

from bert import *

model = BERTModel(vocab_size=1000, hidden_dim=768, max_seq=512, n_layers=12, n_heads=12, dropout=0.1, pad_idx=0)
print(model)

# input: N, T
summary(model, [(1, 512), (1, 512)])
