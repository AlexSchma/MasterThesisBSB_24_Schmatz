import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import os


class ProCNN(nn.Module):
    """
    A two layer CNN that uses max pooling to extract information from
    the sequence input, a one layer fully-connected network to extract TF information
    then trains a fully connected neural network on the to make predictions.

    """

    def __init__(self, config, output_classes):
        super(ProCNN, self).__init__()

        self.config = config

        self.xt_hidden = config["xt_hidden"]
        self.xs_hidden = config["seq_length"] // (config["pool1"] * config["pool2"])

        self.output_classes = output_classes

        self.conv1 = nn.Conv1d(
            config["input_channels"],
            config["conv1_knum"],
            kernel_size=config["conv1_ksize"],
            stride=1,
            padding="same",
        )
        self.pool1 = nn.MaxPool1d(kernel_size=config["pool1"], stride=config["pool1"])
        self.activation1 = self.get_activation(config["conv_activation"])
        self.tanH = nn.Tanh()
        self.conv1dropout = nn.Dropout(config["conv_dropout"])
        self.conv1bn = nn.BatchNorm1d(config["conv1_knum"])
        self.conv2 = nn.Conv1d(
            config["conv1_knum"],
            config["conv2_knum"],
            kernel_size=config["conv2_ksize"],
            stride=1,
            padding="same",
        )
        self.pool2 = nn.MaxPool1d(kernel_size=config["pool2"], stride=config["pool2"])
        self.activation2 = self.get_activation(config["conv_activation"])
        self.conv2dropout = nn.Dropout(config["conv_dropout"])
        self.conv2bn = nn.BatchNorm1d(config["conv2_knum"])
        self.fc_t = nn.Linear(config["n_treatments"], self.xt_hidden)
        self.fc_t_activation = self.get_activation(config["fc_activation"])
        self.fc_t_dropout = nn.Dropout(config["fc_dropout"])

        self.fc1 = nn.Linear(
            config["conv2_knum"] * self.xs_hidden + self.xt_hidden, 512
        )
        self.fc1_activation = self.get_activation(config["fc_activation"])
        self.fc1_dropout = nn.Dropout(config["fc_dropout"])
        self.bn1 = nn.BatchNorm1d(num_features=512)

        self.fc2 = nn.Linear(512, 256)
        self.fc2_activation = self.get_activation(config["fc_activation"])
        self.fc2_dropout = nn.Dropout(config["fc_dropout"])
        self.bn2 = nn.BatchNorm1d(num_features=256)

        self.fc3 = nn.Linear(256, 64)
        self.fc3_activation = self.get_activation(config["fc_activation"])
        self.fc3_dropout = nn.Dropout(config["fc_dropout"])
        self.bn3 = nn.BatchNorm1d(num_features=64)

        self.fc_out = nn.Linear(64, 1)

    @classmethod
    def get_activation(cls, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "selu":
            return nn.SELU()
        else:
            raise ValueError("Invalid activation function: must be relu, gelu, or elu.")

    def forward(self, x_s, x_t):
        x_s = self.conv1(x_s)
        # print("Shape of x_s after conv1: ", x_s.shape)
        x_s = self.pool1(x_s)
        # print("Shape of x_s after pool1: ", x_s.shape)
        x_s = self.activation1(x_s)
        x_s = self.conv1dropout(x_s)
        x_s = self.conv1bn(x_s)
        # x_s = F.scaled_dot_product_attention(x_s, x_s, x_s)

        x_s = self.conv2(x_s)
        # print("Shape of x_s after conv2: ", x_s.shape)
        x_s = self.pool2(x_s)
        # print("Shape of x_s after pool2: ", x_s.shape)
        x_s = self.activation2(x_s)
        x_s = self.conv2dropout(x_s)
        x_s = self.conv2bn(x_s)
        # x_s = F.scaled_dot_product_attention(x_s, x_s, x_s)
        # print("Shape of x_s after conv2: ", x_s.shape)
        x_t = self.fc_t(x_t)
        x_t = self.fc_t_activation(x_t)
        x_t = self.fc_t_dropout(x_t)
        # print("Shape of x_t after fc_t: ", x_t.shape)

        # flatten the conv output
        x_s = torch.flatten(x_s, 1)
        # print("Shape of x_s after flattening: ", x_s.shape)
        # concatenate with the treatment embeddings
        x = torch.cat((x_s, x_t), 1)
        # print("Shape of x after concatenation: ", x.shape)

        x = self.fc1(x)
        x = self.fc1_activation(x)
        x = self.fc1_dropout(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.fc2_activation(x)
        x = self.fc2_dropout(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.fc3_activation(x)
        x = self.fc3_dropout(x)
        x = self.bn3(x)

        x = self.fc_out(x)
        x = self.tanH(x)
        #x = F.softmax(x, dim=1)  # Probs
        x = x.squeeze()
        return x


def get_funprose(
    input_channels=4,
    xt_hidden=1024,
    seq_length=4001,
    conv1_ksize=9,
    conv1_knum=256,
    pool1=20,
    conv2_ksize=13,
    conv2_knum=64,
    pool2=10,
    conv_activation="relu",
    fc_activation="elu",
    conv_dropout=0.3981796388676127,
    tf_dropout=0.18859739941162465,
    fc_dropout=0.016570328292903613,
    n_treatments=3,
):
    config = {
        "input_channels": input_channels,
        "xt_hidden": xt_hidden,
        "seq_length": seq_length,
        "conv1_ksize": conv1_ksize,
        "conv1_knum": conv1_knum,
        "pool1": pool1,
        "conv2_ksize": conv2_ksize,
        "conv2_knum": conv2_knum,
        "pool2": pool2,
        "conv_activation": conv_activation,
        "fc_activation": fc_activation,
        "conv_dropout": conv_dropout,
        "tf_dropout": tf_dropout,
        "fc_dropout": fc_dropout,
        "n_treatments": n_treatments,
    }

    model = ProCNN(config, 2)
    #
    # X = torch.rand(2, 4, 4000)
    # Z = torch.rand(2, 3)

    # model(X, Z)
    return model


if __name__ == "__main__":
    get_funprose()
