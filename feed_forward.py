import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MulticlassFFN(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=64, output_dim=3, dropout_rate=0.1):
        super().__init__()

        self.layer_1 = nn.Linear(embedding_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer_out = nn.Linear(hidden_dim // 2, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x is a batch of BERT embeddings with shape [batch_size, 768].
        # model will have 3 fully connected layers with a relu and dropout layer in-between.

        ff1out = self.layer_1(x)
        relu1out = self.relu(ff1out)
        do_out1 = self.dropout2(relu1out)

        ff2out = self.layer_2(do_out1)
        relu2out = self.relu(ff2out)
        do_out2 = self.dropout(relu2out)

        logits = self.layer_out(do_out2)

        return logits

# Model will recieve the embeddings from Bert and output for Left, Right, or Center
embedding_dim = 768
output_dim = 3

# Changeable parameters
# Changing parameters here will have no effect if running from main
#----------------------
# hidden_dim should be a power of 2
hidden_dim = 64
# Dropout_rate is the precentage of neuron's set to zero for that pass
dropout_rate = 0.1
# Learning rate for the model 
ffn_learning_rate = 5e-4
#-----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





