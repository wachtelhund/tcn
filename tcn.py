import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.dropout1,
                               self.conv2, self.bn2, nn.ReLU(), self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # Ensure the residual has the same sequence length as the output
        if out.size(2) != res.size(2):
            res = F.pad(res, (0, out.size(2) - res.size(2)))
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels=None, kernel_size=None, dropout=None):
        super(TemporalConvNet, self).__init__()
        
        # Use config values if not specified
        num_channels = num_channels or MODEL_CONFIG['num_channels']
        kernel_size = kernel_size or MODEL_CONFIG['kernel_size']
        dropout = dropout or MODEL_CONFIG['dropout']
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size-1) * dilation, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_outputs)

    def forward(self, x):
        # x shape: (batch_size, num_inputs, sequence_length)
        x = self.network(x)
        # x shape: (batch_size, num_channels[-1], sequence_length)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, num_channels[-1])
        x = self.linear(x)  # (batch_size, sequence_length, num_outputs)
        return x 