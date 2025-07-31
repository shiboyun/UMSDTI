import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu(x, dim):
    x = torch.chunk(x, 2, dim=dim)
    return F.silu(x[0]) * x[1]

class SmilesEncoder(nn.Module):
    """smiles feature extraction."""

    def __init__(self, args, vocab_size, hid_dim, n_layers, kernel_size, dropout):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, self.hid_dim, padding_idx=0)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(args.device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])  # convolutional layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(2 * hid_dim) for _ in range(self.n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        # self.ln = nn.LayerNorm(hid_dim)

    def forward(self, smiles):
        conv_input = self.embedding(smiles)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            conved = self.bns[i](conved)
            # conved = [batch size, 2*hid dim, smiles len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = swiglu(conved, dim=1)
            # conved = [batch size, hid dim, smiles len]

            # apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, smiles len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.fc(conved)
        return conved