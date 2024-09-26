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


def attention(query, key):
    """
    Compute 'Scaled Dot Product "Attention"'

    Args:
        query: DNA embedding (batch_size, dim, sequence_length)
        key: hormone embedding (batch_size, dim)
        value: collapsed DNA embedding (batch_size, 1, sequence_length)
    """
    # Calculate the attention weights
    query = query.permute(0, 2, 1)
    key = key.unsqueeze(1)  # Now we have a batch_size x 1 x dim tensor
    # Multiply this with the query tensor
    # This is batch wise matrix multiplication.
    # query is batch_size x sequence_length x dim
    # key(transposed) is batch_size x dim x 1
    # the last dimension of the first tensor should be the same as the second first dimension of the second tensor (i.e. rows x columns)
    attention_wgth = torch.bmm(query, key.transpose(1, 2)) / (query.shape[1] ** 0.5)
    # apply attention to value, elementwise multiplication
    attention_wgth = F.softmax(attention_wgth, dim=1)

    return attention_wgth.permute(0, 2, 1)


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


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = my_conv1d(input_dim, output_dim, kernel_size, stride, padding)
        self.conv2 = my_conv1d(output_dim, output_dim, kernel_size, stride, padding)
        if input_dim != output_dim:
            self.downsample = nn.Conv1d(
                input_dim, output_dim, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))  # First convolution
        out = self.conv2(out)  # Second convolution
        if residual.size(1) != out.size(1):
            residual = self.downsample(residual)
        out = out + residual  # Adding residual connection
        out = F.relu(out)
        return out


class ResidualCNN(nn.Module):
    def __init__(
        self,
        core_input_size,
        conditional_input_size,
        sequence_length,
        max_pooling_kernel_size,
        residual_dim,
        num_classes,
        num_residual_blocks,
    ):
        super(ResidualCNN, self).__init__()
        self.conv1 = my_conv1d(
            core_input_size, residual_dim, kernel_size=3, stride=1, padding="same"
        )
        self.conditional_input_embedding = conditional_embedding(
            conditional_input_size, core_input_size
        )

        self.conditional_embedding_adaptator = nn.ModuleList(
            [
                nn.Linear(core_input_size, residual_dim * (1 + (i // 5)))
                for i in range(1, num_residual_blocks)
            ]
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    input_dim=residual_dim * (1 + ((i - 1) // 5)),
                    output_dim=residual_dim * (1 + (i // 5)),
                    kernel_size=3,
                    stride=1,
                    padding="same",
                )
                for i in range(1, num_residual_blocks)
            ]
        )

        self.downsample = nn.Linear(
            # residual_dim * (1 + ((num_residual_blocks-1) // 5)) * int(sequence_length/max_pooling_kernel_size), num_classes
            residual_dim * (1 + ((num_residual_blocks - 1) // 5)),
            64,
        )
        self.ffn = nn.ModuleList([nn.Linear(64, 128), nn.Linear(128, 2)])

        self.max_pooling_kernel_size = max_pooling_kernel_size

    def forward(self, x, z):
        embedding_condition = self.conditional_input_embedding(z)
        x = x + embedding_condition.unsqueeze(2)
        x = F.relu(self.conv1(x))
        for block in zip(self.residual_blocks, self.conditional_embedding_adaptator):
            embedding_condition_adapted = block[1](embedding_condition)
            processed_x = block[0](x)
            x = processed_x + embedding_condition_adapted.unsqueeze(2)
        # x = F.max_pool1d(x, self.max_pooling_kernel_size, self.max_pooling_kernel_size)
        x = F.avg_pool1d(x, x.size(2))  # Global average pooling
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.3)
        x = self.downsample(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        x = self.ffn[0](x)
        x = F.relu(x)
        x = self.ffn[1](x)
        x = F.softmax(x, dim=1)  # Softmax for binary classification
        return x


def get_cnn(reload, device):
    # If model does not exist, create it
    if not os.path.exists("Code/Models/CNNs/ResidualCNN_model.pth"):
        model = ResidualCNN(4, 3, 4001, 6, 64, 2, 12).to(device)
    elif reload:
        model = ResidualCNN(4, 3, 4001, 6, 64, 2, 12).to(device)
        model.load_state_dict(torch.load("Code/Models/CNNs/ResidualCNN_model.pth"))
        print("Model loaded")
    else:
        model = ResidualCNN(4, 3, 4001, 6, 64, 2, 12).to(device)
    return model
