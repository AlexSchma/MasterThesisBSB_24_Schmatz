import argparse
import os
import sys

# sys.path.append('C:/Users/alexa/OneDrive/Dokumente/GitHub/DL-GRN')
# print(os.getcwd())
import pandas as pd
from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset, DataLoader

# from Code.DataParsing.DataUtils import family_wise_train_test_splitting
import pickle
import numpy as np
#from data import PandasDataset
from datetime import datetime
from sklearn.metrics import matthews_corrcoef

from sklearn.model_selection import GroupShuffleSplit, GroupKFold


def family_wise_train_test_splitting(
    X: np.array,
    Y: np.array,
    gene_families: np.array,
    test_size=0.2,
    random_state=42,
    return_index=False,
):
    """
    Split the data into training and testing sets, ensuring that the gene families are not split between the two sets.

    X: The input data
    Y: The labels
    gene_families: The gene families
    test_size: The proportion of the data to be used as the test set
    random_state: The random state for the shuffle split
    return_index: If True, return the indices of the split instead of the data
    ---
    returns: The training and testing sets:
        X_train, X_test, Y_train, Y_test, gene_families_train, gene_families_test

    """
    # assert gene_id and Y are 1 dimensional (1 column)
    assert len(X.shape) == 2
    assert len(Y.shape) == 1
    assert len(gene_families.shape) == 1

    # Create the split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Split the data
    train_index, test_index = next(gss.split(X, Y, groups=gene_families))

    # assert no family is present in both sets
    assert (
        len(
            set(gene_families[train_index]).intersection(set(gene_families[test_index]))
        )
        == 0
    )

    # Return the split data
    if return_index:
        return train_index, test_index
    else:
        return (
            X[train_index],
            X[test_index],
            Y[train_index],
            Y[test_index],
            gene_families[train_index],
            gene_families[test_index],
        )


base_dir = "FunProse_Results"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
logging_dir = os.path.join(base_dir, f"Run_{timestamp}")
os.makedirs(logging_dir, exist_ok=True)
model_path = os.path.join(logging_dir, "model.pth")
training_log_path = os.path.join(logging_dir, "training_log.pkl")

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Run single classifier for FUN-PROSE")
parser.add_argument("-use_cuda", type=bool, default=False)

parser.add_argument("--test", action="store_true", help="For testing e.g. on laptop")
# Parse the arguments
args = parser.parse_args()

# Use the boolean argument
if args.test:
    print("Test mode is on")
else:
    print("Regular mode")

# Load XY format one-hot encoded data
with open("Data/Processed/XY_formats/One_hot_encoded.pkl", "rb") as f:
    xy_dict = pickle.load(f)

# load the gene names
gene_names = np.load("Data/Processed/XY_formats/gene_names_one_hot.npy")

# Select randomly 30000 instances index
# np.random.seed(7215)
# random_index = np.random.choice(range(len(gene_names)), 10000, replace=False)

X_sequence = np.array(
    [xy_dict[gene][0][0].T for gene in xy_dict]
).squeeze()  # [random_index]
X_treatment = np.array([xy_dict[gene][0][1] for gene in xy_dict])  # [random_index]
Y = np.array([xy_dict[gene][1] for gene in xy_dict])  # [random_index]

# Get family names per gene
# load gene families
gene_families = pd.read_csv("Data/Processed/gene_families.csv", index_col=1)

# Map gene names to family names
# I cannot use a dictionary because the genes are repeated 5 times, eachper treatment
# I will use a list instead
family_names = []
for gene in gene_names:
    family_names.append(gene_families.loc[gene, "family_id"])


# Split the data

# train_index, test_index = family_wise_train_test_splitting(X_treatment, Y, np.array(family_names)[random_index], random_state = 7215, return_index=True)
train_index, test_index = family_wise_train_test_splitting(
    X_treatment, Y, np.array(family_names), random_state=7215, return_index=True
)
# Get valiadation set
train_index, val_index = family_wise_train_test_splitting(
    X_treatment[train_index],
    Y[train_index],
    np.array(family_names)[train_index],
    random_state=7215,
    return_index=True,
)
# X_sequence = torch.tensor(X_sequence, dtype=torch.float32)
# X_treatment = torch.tensor(X_treatment, dtype=torch.float32)
# Y = torch.tensor(Y, dtype=torch.long)
X_train_sequence = X_sequence[train_index]
X_test_sequence = X_sequence[test_index]
X_train_treatment = X_treatment[train_index]
X_test_treatment = X_treatment[test_index]
Y_train = Y[train_index]
Y_test = Y[test_index]
X_val_sequence = X_sequence[val_index]
X_val_treatment = X_treatment[val_index]
Y_val = Y[val_index]

if args.test:
    X_train_sequence = X_train_sequence[:16]
    X_test_sequence = X_test_sequence[:16]
    X_train_treatment = X_train_treatment[:16]
    X_test_treatment = X_test_treatment[:16]
    Y_test = Y_test[:16]
    X_val_sequence = X_val_sequence[:16]
    X_val_treatment = X_val_treatment[:16]
    Y_val = Y_val[:16]


class Vocab:
    """
    A simple vocabulary class that takes an alphabet of symbols,
    and assigns each symbol a unique integer id., e.g.

    > v = Vocab(['a','b','c','d'])
    > v('c')
    2
    > v('a')
    0
    > len(v)
    4

    """

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.index_map = {
            letter: index for (index, letter) in list(enumerate(alphabet))
        }

    def __len__(self):
        return len(self.alphabet)

    def __call__(self, letter):
        return self.index_map[letter]


class Tensorize:
    """
    An instance of Tensorize is a function that maps a piece of data
    (i.e. a dictionary) to an input and output tensor for consumption by
    a neural network.
    """

    def __init__(self, symbol_vocab, max_word_length, cudaify):
        self.symbol_vocab = symbol_vocab
        self.max_word_length = max_word_length
        self.cudaify = cudaify

    def __call__(self, data):
        words = Tensorize.words_to_tensor(
            data["seq"], self.symbol_vocab, self.max_word_length
        ).float()

        tfs = torch.stack(data["tfs"], dim=1).float()
        label = data["exp"].float()
        return self.cudaify(words), self.cudaify(tfs), self.cudaify(label)

    @staticmethod
    def words_to_tensor(words, vocab, max_word_length):
        """
        Turns an K-length list of words into a <K, len(vocab), max_word_length>
        tensor.

        e.g.
            t = words_to_tensor(['BAD', 'GAB'], Vocab('ABCDEFG'), 3)
            # t[0] is a matrix representations of 'BAD', where the jth
            # column is a one-hot vector for the jth letter
            print(t[0])

        """
        tensor = torch.zeros(len(words), len(vocab), max_word_length)
        for i, word in enumerate(words):
            start_index = max(0, len(word) - max_word_length)
            for li, letter in enumerate(word[start_index : len(word)][::-1]):
                tensor[i][vocab(letter)][max_word_length - li - 1] = 1
        return tensor


class ProCNN(torch.nn.Module):
    """
    A two layer CNN that uses max pooling to extract information from
    the sequence input, a one layer fully-connected network to extract TF information
    then trains a fully connected neural network on the to make predictions.

    """

    def __init__(self, config, output_classes, input_symbol_vocab):
        super(ProCNN, self).__init__()

        self.config = config
        self.input_symbol_vocab = input_symbol_vocab

        self.xt_hidden = config["xt_hidden"]
        self.xs_hidden = config["seq_length"] // (config["pool1"] * config["pool2"])

        self.output_classes = output_classes

        self.conv1 = torch.nn.Conv1d(
            4,
            config["conv1_knum"],
            kernel_size=config["conv1_ksize"],
            stride=1,
            padding=config["conv1_ksize"] // 2,
        )
        self.pool1 = torch.nn.MaxPool1d(
            kernel_size=config["pool1"], stride=config["pool1"]
        )
        self.activation1 = self.get_activation(config["conv_activation"])
        self.conv1dropout = torch.nn.Dropout(config["conv_dropout"])
        self.conv1bn = torch.nn.BatchNorm1d(config["conv1_knum"])

        self.conv2 = torch.nn.Conv1d(
            config["conv1_knum"],
            config["conv2_knum"],
            kernel_size=config["conv2_ksize"],
            stride=1,
            padding=config["conv2_ksize"] // 2,
        )
        self.pool2 = torch.nn.MaxPool1d(
            kernel_size=config["pool2"], stride=config["pool2"]
        )
        self.activation2 = self.get_activation(config["conv_activation"])
        self.conv2dropout = torch.nn.Dropout(config["conv_dropout"])
        self.conv2bn = torch.nn.BatchNorm1d(config["conv2_knum"])

        self.fc_t = torch.nn.Linear(config["num_tfs"], self.xt_hidden)
        self.fc_t_activation = self.get_activation(config["fc_activation"])
        self.fc_t_dropout = torch.nn.Dropout(config["fc_dropout"])

        self.fc1 = torch.nn.Linear(
            config["conv2_knum"] * self.xs_hidden + self.xt_hidden, 512
        )
        self.fc1_activation = self.get_activation(config["fc_activation"])
        self.fc1_dropout = torch.nn.Dropout(config["fc_dropout"])
        self.bn1 = torch.nn.BatchNorm1d(num_features=512)

        self.fc2 = torch.nn.Linear(512, 256)
        self.fc2_activation = self.get_activation(config["fc_activation"])
        self.fc2_dropout = torch.nn.Dropout(config["fc_dropout"])
        self.bn2 = torch.nn.BatchNorm1d(num_features=256)

        self.fc3 = torch.nn.Linear(256, 64)
        self.fc3_activation = self.get_activation(config["fc_activation"])
        self.fc3_dropout = torch.nn.Dropout(config["fc_dropout"])
        self.bn3 = torch.nn.BatchNorm1d(num_features=64)

        self.fc_out = torch.nn.Linear(64, 1)

    @classmethod
    def get_activation(cls, activation):
        if activation == "relu":
            return torch.nn.ReLU()
        elif activation == "gelu":
            return torch.nn.GELU()
        elif activation == "elu":
            return torch.nn.ELU()
        elif activation == "selu":
            return torch.nn.SELU()
        else:
            raise ValueError("Invalid activation function: must be relu, gelu, or elu.")

    def get_input_vocab(self):
        return self.input_symbol_vocab

    def forward(self, x_s, x_t):
        b = list(x_s.size())[0]  # batch size
        x_s = self.conv1(x_s)
        x_s = self.pool1(x_s)
        x_s = self.activation1(x_s)
        x_s = self.conv1dropout(x_s)
        x_s = self.conv1bn(x_s)

        x_s = self.conv2(x_s)
        x_s = self.pool2(x_s)
        x_s = self.activation2(x_s)
        x_s = self.conv2dropout(x_s)
        x_s = self.conv2bn(x_s)

        x_t = self.fc_t(x_t)
        x_t = self.fc_t_activation(x_t)
        x_t = self.fc_t_dropout(x_t)

        x = torch.cat(
            (
                x_s.reshape((b, self.config["conv2_knum"] * self.xs_hidden, 1)).view(
                    -1, self.config["conv2_knum"] * self.xs_hidden
                ),
                x_t,
            ),
            1,
        )
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
        return x


class Trainable:
    # def __init__(self):
    #     self.best_val_loss = float('inf')  # Initialize best validation loss

    class MockScaler:
        def scale(self, x):
            return x

        def step(self, optim):
            optim.step()

        def update(self):
            pass

    def setup(self, config, output_classes, char_vocab, trainset, devset):
        """
        Initializes a trainer to train a neural network using the provided DataLoaders
        (for training and development data).

        tensorize is a function that maps a piece of data (i.e. a dictionary)
        to an input and output tensor for the neural network.

        char_vocab is the Vocab object of the input data.

        """
        print("Setting up Trainable")
        self.config = config
        self.char_vocab = char_vocab

        # self.train_loader = DataLoader(trainset, batch_size=config["batch_size"], num_workers=4, pin_memory=True, shuffle=True)
        # self.dev_loader = DataLoader(devset, batch_size=config["batch_size"], num_workers=4, pin_memory=True, shuffle=False)
        def custom_collate(batch):
            batch_X = np.stack([sample["X"] for sample in batch])
            batch_Z = np.stack([sample["Z"] for sample in batch])
            batch_Y = np.stack([sample["Y"] for sample in batch])
            return {"X": batch_X, "Z": batch_Z, "Y": batch_Y}

        self.train_loader = DataLoader(
            trainset,
            batch_size=config["batch_size"],
            collate_fn=custom_collate,
            shuffle=True,
        )
        self.dev_loader = DataLoader(
            devset,
            batch_size=config["batch_size"],
            collate_fn=custom_collate,
            shuffle=True,
        )
        self.best_dev_corr = 0

        self.use_cuda = config["use_cuda"]
        if self.use_cuda:
            self.cudaify_model = lambda x: x.cuda()
            self.cudaify = lambda x: x.cuda(non_blocking=True)
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.cudaify_model = lambda x: x
            self.cudaify = lambda x: x
            self.scaler = MockScaler()

        self.tensorize = Tensorize(char_vocab, config["seq_length"], self.cudaify)
        self.net = self.cudaify_model(ProCNN(config, output_classes, char_vocab))
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        # self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.terminate = False

    def cleanup(self):
        self.terminate = True

        del self.config
        del self.tensorize
        del self.char_vocab
        del self.train_loader
        del self.dev_loader

        del self.best_dev_corr

        del self.use_cuda
        del self.cudaify_model
        del self.cudaify
        del self.scaler

        del self.net
        del self.optimizer
        del self.loss

        torch.cuda.empty_cache()

    @classmethod
    def correlation(self, x, y):
        return pearsonr(x, y)[0]

    def step(self):
        self.net.train()
        train_loss = self.run_one_cycle(train=True)
        self.net.eval()
        with torch.no_grad():
            dev_loss, dev_corr = self.run_one_cycle(train=False)
            self.best_dev_corr = max(self.best_dev_corr, dev_corr)

        return {
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            # "dev_corr" : dev_corr,
            # "best_dev_corr" : self.best_dev_corr,
            "MCC": dev_corr,
            "best_MCC": self.best_dev_corr,
        }

    def run_one_cycle(self, train=True):
        total_loss = 0

        if not train:
            all_labels = []
            all_preds = []

        for i, data in enumerate(self.train_loader if train else self.dev_loader):
            if self.terminate:
                return None
            if i % 10 == 0:
                pass
            # print(i, "/", len(self.train_loader if train else self.dev_loader))
            if train:
                self.optimizer.zero_grad()

            # input_s, input_t, labels = self.tensorize(data)
            # batch_X = torch.tensor(data['X'], dtype = torch.float32).to("cuda")  # Batch of X sequence
            batch_X = torch.tensor(data["X"][:, :, 2000:3000], dtype=torch.float32).to(
                "cuda"
            )  # Batch of X sequence
            batch_Z = torch.tensor(data["Z"], dtype=torch.float32).to(
                "cuda"
            )  # Batch of X condition
            labels = torch.tensor(data["Y"], dtype=torch.float32).to(
                "cuda"
            )  # Batch of label
            outputs = self.net(batch_X, batch_Z).view(-1)
            if self.use_cuda:
                with torch.cuda.amp.autocast():
                    loss = self.loss(outputs, labels)
            else:
                loss = self.loss(outputs, labels)

            if not train:
                all_labels.extend(labels.detach().tolist())
                all_preds.extend(outputs.detach().tolist())
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.detach().item()

        total_loss = total_loss / len(self.train_loader if train else self.dev_loader)
        if not train:
            # total_corr = self.correlation(all_labels, all_preds)

            binary_preds = [1 if pred >= 0.5 else 0 for pred in all_preds]
            # binary_labels = [int(value) for value in all_labels]
            # print(binary_labels)
            # print(binary_preds)
            MCC = matthews_corrcoef(all_labels, binary_preds)
            # return total_loss, total_corr
            return total_loss, MCC

        return total_loss

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint.pt")

        torch.save(
            {
                "best_dev_corr": self.best_dev_corr,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        path = os.path.join(checkpoint_path, "checkpoint.pt")
        state = torch.load(path)

        self.best_dev_corr = state["best_dev_corr"]
        self.net.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])


def run_single_model(args, mode_config, output_classes, trainset, devset):
    """
    Performs a hyperparameter search of the best-performing network
    for the given datasets (training and development).
    """

    def extract_amino_acid_alphabet(trainset):
        return ["A", "C", "G", "N", "T"]

    char_vocab = Vocab(extract_amino_acid_alphabet(trainset))

    trainer = Trainable()
    trainer.setup(
        config=config,
        char_vocab=char_vocab,
        output_classes=output_classes,
        trainset=trainset,
        devset=devset,
    )

    # epoch = 1
    # while True:
    #     print(f"Epoch {epoch} ", trainer.step())
    #     epoch += 1

    # return results
    epoch = 0
    epoch_logging = {}
    while True:
        if epoch == 500:
            break
        epoch_logging[epoch] = trainer.step()
        print(f"Epoch {epoch} ", epoch_logging[epoch])
        with open(training_log_path, "wb") as f:
            pickle.dump(epoch_logging, f)
        # save the model
        # torch.save(model.state_dict(), "Code/Models/CNNs/ResidualCNN_model.pth")
        torch.save(trainer.net.state_dict(), model_path)
        epoch += 1


if __name__ == "__main__":
    config = {
        "learning_rate": 0.00010229218879330196,
        "weight_decay": 0.0016447149582678627,
        "xt_hidden": 1024,
        "seq_length": 1000,
        "conv1_ksize": 9,
        "conv1_knum": 256,
        "pool1": 19,
        "conv2_ksize": 13,
        "conv2_knum": 64,
        "pool2": 8,
        "batch_size": 256,
        "conv_activation": "relu",
        "fc_activation": "elu",
        "conv_dropout": 0.3981796388676127,
        "tf_dropout": 0.18859739941162465,
        "fc_dropout": 0.016570328292903613,
        "use_cuda": args.use_cuda,
        "num_tfs": 3,
    }

    class CustomDataset(Dataset):
        def __init__(self, X, Z, Y, transform=None):
            self.X = X
            self.Z = Z
            self.Y = Y
            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            sample = {"X": self.X[idx], "Z": self.Z[idx], "Y": self.Y[idx]}

            if self.transform:
                sample = self.transform(sample)

            return sample

    print("Loading data")
    # trainset = PandasDataset('/home/simonl2/yeast/Gene_Expression_Pred/October_Runs/Data/file_oav_filt05_filtcv3_Zlog_new_trainGenes.pkl')
    # devset = PandasDataset('/home/simonl2/yeast/Gene_Expression_Pred/October_Runs/Data/file_oav_filt05_filtcv3_Zlog_new_validGenes.pkl')
    trainset = CustomDataset(X_train_sequence, X_train_treatment, Y_train)
    devset = CustomDataset(X_val_sequence, X_val_treatment, Y_val)

    print("Loaded data")

    run_single_model(args, config, 1, trainset, devset)
