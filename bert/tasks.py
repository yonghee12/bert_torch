import torch.nn as nn
import torch.nn.functional as F


class MaskedLanguageModelTask(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()
        self.logit = nn.Linear(hidden_dim, vocab_size)

    def forward(self, out):
        return F.log_softmax(self.logit(out), dim=-1)


class NextSentencePredictionTask(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.logit = nn.Linear(hidden_dim, 2)

    def forward(self, out):
        return F.log_softmax(self.logit(out[:, 0, :]), dim=-1)
