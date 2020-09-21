from torchsummary import summary

from bert import *

model = BERTModel(vocab_size=10000, hidden_dim=768, max_seq=512, n_layers=12, n_heads=12, dropout=0.1, pad_idx=0)
print(model)

summary(model, (512, 768))