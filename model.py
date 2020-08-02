import torch.nn as nn
from torch.utils.data import Dataset


class Data(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class MLP(nn.Module):

    def __init__(self, n_neurons, dropout, batch_norm, activation):

        super(MLP, self).__init__()

        self.n_neurons = n_neurons
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation

        self.model = []
        self._init_params()

    def _init_params(self):

        if self.dropout:
            self.model.append(nn.Dropout(self.dropout))
        for idx in range(len(self.n_neurons)):
            self.model.append(nn.Linear(self.n_neurons[idx][0], self.n_neurons[idx][1]))
            if idx != len(self.n_neurons) - 1:
                self.model.append(self.activation)
                if self.batch_norm:
                    self.model.append(nn.BatchNorm1d(self.n_neurons[idx][1]))

        self.model = nn.ModuleList(self.model)

    def forward(self, x):

        for layer in self.model:
            x = layer(x)

        return x