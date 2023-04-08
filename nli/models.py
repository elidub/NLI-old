import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_length = 100

    def forward(self, embedding, length):

        assert embedding.shape == (length.shape[0], self.max_length, 300)

        mask = (length.unsqueeze(1) > torch.arange(embedding.shape[1], device=embedding.device)).float().unsqueeze(2)
        mask = mask.to(embedding.device)

        embedding_sum = torch.sum(embedding * mask, dim = 1)
        length_sum = torch.sum(mask, dim = 1)
        mean = embedding_sum / length_sum

        return mean

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)