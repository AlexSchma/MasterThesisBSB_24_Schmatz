import torch
import torch.nn as nn
import torch.nn.functional as F


# The idea here is to construct a very very simple CNN.


class CNNmodule(nn.Module):
    def __init__(self, n_layers_before:int, kernel_sizes_before:list, n_filters_before:int, local_pooling_size_1:int,
                        n_layers_mid:int, kernel_sizes_mid:list, n_filters_mid:int, local_pooling_size_2:int, 
                        n_layers_last:int, kernel_sizes_last:list, n_filters_last:int, dropout=0.5):
        super(CNNmodule, self).__init__()
        ## The idea is to have equivariant layer, local pooling layer, equivariant layer and global pooling layer.

        first_CNNs = []
        for i in range(n_layers_before):
            if i == 0:
                first_CNNs.append(nn.Conv1d(in_channels=4, out_channels=n_filters_before, kernel_size=kernel_sizes_before[i]))
            else:
                first_CNNs.append(nn.Conv1d(in_channels=n_filters_before, out_channels=n_filters_before, kernel_size=kernel_sizes_before[i]))
            # batch normalization
            first_CNNs.append(nn.BatchNorm1d(n_filters_before))
            first_CNNs.append(nn.ReLU())
        
        self.first_CNNs = nn.Sequential(*first_CNNs)

        self.local_pooling1 = nn.MaxPool1d(kernel_size=local_pooling_size_1, stride=local_pooling_size_1)

        second_CNNs = []
        for i in range(n_layers_mid):
            if i == 0:
                second_CNNs.append(nn.Conv1d(in_channels=n_filters_before, out_channels=n_filters_mid, kernel_size=kernel_sizes_mid[i]))
            else:
                second_CNNs.append(nn.Conv1d(in_channels=n_filters_mid, out_channels=n_filters_mid, kernel_size=kernel_sizes_mid[i]))
            # batch normalization
            second_CNNs.append(nn.BatchNorm1d(n_filters_mid))
            second_CNNs.append(nn.ReLU())

        self.second_CNNs = nn.Sequential(*second_CNNs)

        self.local_pooling2 = nn.MaxPool1d(kernel_size=local_pooling_size_2, stride=local_pooling_size_2)

        third_CNNs = []
        for i in range(n_layers_last):
            if i == 0:
                third_CNNs.append(nn.Conv1d(in_channels=n_filters_mid, out_channels=n_filters_last, kernel_size=kernel_sizes_last[i]))
            else:
                third_CNNs.append(nn.Conv1d(in_channels=n_filters_last, out_channels=n_filters_last, kernel_size=kernel_sizes_last[i]))
            # batch normalization
            third_CNNs.append(nn.BatchNorm1d(n_filters_last))
            third_CNNs.append(nn.ReLU())
        
        self.third_CNNs = nn.Sequential(*third_CNNs)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.first_CNNs(x)
        x = self.dropout(x)
        x = self.local_pooling1(x)
        x = self.second_CNNs(x)
        x = self.dropout(x)
        x = self.local_pooling2(x)
        x = self.third_CNNs(x)
        x = self.dropout(x)
        x = self.global_pooling(x)
        return x
    
class embeding_module(nn.Module):
    def __init__(self, embedding_input_size:int, embedding_output_size:int):
        super(embeding_module, self).__init__()
        self.embedding = nn.Linear(embedding_input_size, embedding_output_size)

    def forward(self, x):
        x = self.embedding(x)
        return x
    

class FNN_module(nn.Module):
    def __init__(self, n_layers:int, input_size:int, hidden_sizes:list, output_size:int):
        super(FNN_module, self).__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    


class simple_cnn_ffn(nn.Module):
    def __init__(self, config):
        super(simple_cnn_ffn, self).__init__()
        self.config = config
        self.CNN = CNNmodule(config["n_layers_before"], config["kernel_sizes_before"], config["n_filters_before"], config["local_pooling1_size"],
                            config["n_layers_mid"], config["kernel_sizes_mid"], config["n_filters_mid"], config["local_pooling2_size"],
                            config["n_layers_last"], config["kernel_sizes_last"], config["n_filters_last"])
        

        self.embedding = embeding_module(config["embedding_input_size"], config["embedding_output_size"])
        self.FNN = FNN_module(config["n_layers"], config["input_size"], config["hidden_sizes"], config["output_size"])

    def forward(self, x_s, x_t):
        x_s = self.CNN(x_s)
        # remove dimensions of size 1
        x_s = x_s.flatten(start_dim=1)
        x_t = self.embedding(x_t)
        # Concatenate the CNN output with the embedding output, must have the same number of channels, both have length 1
        x = torch.cat((x_s, x_t), dim=1)
        x = self.FNN(x)
        return x
    

#class triple_head_cnn(nn.Module):
#    '''
#    Three input heads for TSS, introns and TTS.
#    '''
#    pass
#
def generate_model(n_layers_before=2, kernel_sizes_before=[15, 15], n_filters_before=64, local_pooling_size1=10,
                    n_layers_mid=2, kernel_sizes_mid=[9, 9], n_filters_mid=64, local_pooling_size2=2,
                    n_layers_last=2, kernel_sizes_last=[3, 3], n_filters_last=64, embedding_input_size=3,
                    embedding_output_size=64, n_layers=2, hidden_sizes=[64] * 4, output_size = 2):
    
    input_size = n_filters_mid + embedding_output_size
    config = {
        "n_layers_before": n_layers_before,
        "kernel_sizes_before": kernel_sizes_before,
        "n_filters_before": n_filters_before,
        "local_pooling1_size": local_pooling_size1,
        "n_layers_mid": n_layers_mid,
        "kernel_sizes_mid": kernel_sizes_mid,
        "n_filters_mid": n_filters_mid,
        "local_pooling2_size": local_pooling_size2,
        "n_layers_last": n_layers_last,
        "kernel_sizes_last": kernel_sizes_last,
        "n_filters_last": n_filters_last,
        "embedding_input_size": embedding_input_size,
        "embedding_output_size": embedding_output_size,
        "n_layers": n_layers,
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size
    }
    model = simple_cnn_ffn(config)
    return model




if __name__ == "__main__":
    model = generate_model()
    print(model)
    x_s = torch.rand(1, 4, 1000)
    x_t = torch.rand(1, 3)
    print(model(x_s, x_t))
    print(model(x_s, x_t).shape)
    print("Done!")