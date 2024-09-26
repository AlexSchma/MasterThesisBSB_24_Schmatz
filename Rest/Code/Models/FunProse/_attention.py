import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import os


class my_conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(my_conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


class conditional_embedding(nn.Module):
    def __init__(self, input_size, output_size):
        super(conditional_embedding, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def attention(query, key, value):
    """
    Compute 'Scaled Dot Product "Attention"'

    Args:
        query: DNA embedding (batch_size, dim, sequence_length)
        key: hormone embedding (batch_size, dim)
        value: collapsed DNA embedding (batch_size, 1, sequence_length)
    """
    # Calculate the attention weights
    query = query.permute(0, 2, 1)
    value = value.permute(0, 2, 1)
    key = key.unsqueeze(1)  # Now we have a batch_size x 1 x dim tensor
    # Multiply this with the query tensor
    # This is batch wise matrix multiplication.
    # query is batch_size x sequence_length x dim
    # key(transposed) is batch_size x dim x 1
    # the last dimension of the first tensor should be the same as the second first dimension of the second tensor (i.e. rows x columns)
    attention_wgth = torch.bmm(query, key.transpose(1, 2)) / (query.shape[1] ** 0.5)
    # apply attention to value, elementwise multiplication
    attention_wgth = F.softmax(attention_wgth, dim=1)
    x = attention_wgth * value

    return x.permute(0, 2, 1)


class attention_block(nn.Module):
    def __init__(
        self, input_dim_sequence, input_dim_embedding, output_dim, kernel_size=6
    ):
        super(attention_block, self).__init__()
        self.conv = my_conv1d(input_dim_sequence, output_dim, kernel_size, 1, "same")
        self.embedding = conditional_embedding(input_dim_embedding, output_dim)
        self.value_conv = my_conv1d(output_dim, 1, kernel_size, 1, "same")

    def forward(self, x, y):
        x = self.conv(x)
        y = self.embedding(y)
        x = attention(x, y, self.value_conv(x))
        return x


class attention_net(nn.Module):
    def __init__(
        self,
        input_dim_sequence,
        input_dim_embedding,
        sequene_length,
        output_dim,
        ff_dim=128,
        blocks=5,
        kernel_size=6,
        max_pooling_stride=6,
        max_pooling_kernel_size=6,
    ):
        super(attention_net, self).__init__()

        self.attention_blocks = nn.ModuleList(
            [
                attention_block(
                    input_dim_sequence if i == 0 else 1,
                    input_dim_embedding,
                    output_dim,
                    kernel_size,
                )
                for i in range(blocks)
            ]
        )
        self.ff1 = nn.Linear(int(sequene_length / max_pooling_stride), ff_dim)
        self.ff2 = nn.Linear(ff_dim, ff_dim)
        self.ff3 = nn.Linear(ff_dim, 2)
        self.max_pooling_stride = max_pooling_stride
        self.max_pooling_kernel_size = max_pooling_kernel_size

    def forward(self, x, y):
        residual = 0
        for i in range(len(self.attention_blocks)):
            x = self.attention_blocks[i](x, y)
            x += residual
            residual = x
        x = F.max_pool1d(x, self.max_pooling_kernel_size, self.max_pooling_stride)
        x = x.squeeze()
        x = F.relu(self.ff1(x))
        x = F.relu(self.ff2(x))
        x = self.ff3(x)
        x = F.softmax(x, dim=1)
        return x


def get_attention_net(
    input_dim_sequence,
    input_dim_embedding,
    sequene_length,
    output_dim,
    ff_dim=128,
    blocks=5,
    kernel_size=6,
    max_pooling_stride=6,
    max_pooling_kernel_size=6,
):
    return attention_net(
        input_dim_sequence,
        input_dim_embedding,
        sequene_length,
        output_dim,
        ff_dim=ff_dim,
        blocks=blocks,
        kernel_size=kernel_size,
        max_pooling_stride=max_pooling_stride,
        max_pooling_kernel_size=max_pooling_kernel_size,
    )
