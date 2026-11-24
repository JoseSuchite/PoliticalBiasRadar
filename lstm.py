import torch
import torch.nn as nn

# BERT Embedding dims = 768
# Many to Many LSTM

class MulticlassLSTM(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=64,
        output_dim=3,
        n_layers=1,
        dropout_rate=0.15,
        bidirectional=False,
        projection_dim=None
    ):
        super().__init__()

        projection_dim = hidden_dim if projection_dim is None else projection_dim
        self.projection = nn.Linear(embedding_dim, projection_dim)

        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout_rate)

        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * directions, output_dim)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError("Expected [batch, timestep, embedding]")

        projected = torch.relu(self.projection(x))

        out, (hidden, cell) = self.lstm(projected)

        # One classification per time step
        out = self.dropout(out)

        logits = self.fc(out)

        return logits