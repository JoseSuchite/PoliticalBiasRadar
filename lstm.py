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
        dropout_rate=0.15
    ):
        super().__init__()

        self.projection = nn.Linear(
            embedding_dim, hidden_dim
        )

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError("Expected [batch, timestep, embedding]")

        projected = torch.relu(self.projection(x))

        out, (hidden, cell) = self.lstm(projected)

        # One classification per time step
        out = self.dropout(out)

        logits = self.fc(out)

        return logits