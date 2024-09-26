import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.2,seq_length=100):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Dropout layer after convolutional layers
        self.conv_dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(32 * (seq_length // 2 // 2), 64)  # Adjust based on your sequence length
        # Dropout layer before the final fully connected layer
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, 1)  # Assuming binary classification

    def forward(self, x,seq_length=100):
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = self.conv_dropout(x)  # Apply dropout after first conv layer
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = self.conv_dropout(x)  # Apply dropout after second conv layer
        x = x.view(-1, 32 * (seq_length // 2 // 2))  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)  # Apply dropout before final fully connected layer
        x = self.fc2(x)
        return x

class ModifiedSimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5,seq_length=100):  # Add a dropout_rate parameter with a default value
        super(ModifiedSimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Dropout layer after convolutional layers
        self.conv_dropout = nn.Dropout(p=dropout_rate)

        
        # Calculate the new sequence length after conv and pooling layers
        new_seq_length = seq_length // 5  # After first max pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * new_seq_length, 64)  # Adjust the multiplication factor according to your conv2 output
        self.fc2 = nn.Linear(64, 1)  # Assuming binary classification
        
        # Dropout layer before the final fully connected layer
        self.fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x,seq_length=100):
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 5)  # Adjusted max_pool1d size to 5
        x = self.conv_dropout(x)  # Apply dropout after first conv layer
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = self.conv_dropout(x)  # Apply dropout after second conv layer
        x = x.view(-1, 32 * (seq_length // 5 // 2))  # Adjust the division factor according to your new sequence length
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)  # Apply dropout before final fully connected layer
        x = self.fc2(x)
        return x

class ModifiedSimpleCNN_cond(nn.Module):
    def __init__(self, dropout_rate=0.5,seq_length=100):  # Add a dropout_rate parameter with a default value
        super(ModifiedSimpleCNN_cond, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Dropout layer after convolutional layers
        self.conv_dropout = nn.Dropout(p=dropout_rate)

        # Assuming the sequence length of your input is in X_train.shape[1]
        
        # Calculate the new sequence length after conv and pooling layers
        new_seq_length = seq_length // 5  # After first max pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * new_seq_length + 64, 64)  # Adjust the multiplication factor according to your conv2 output
        self.fc2 = nn.Linear(64, 1)  # Assuming binary classification
        self.fc1c = nn.Linear(1,32)
        self.fc2c = nn.Linear(32,64)
        # Dropout layer before the final fully connected layer
        self.fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x,z,seq_length=100):
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 5)  # Adjusted max_pool1d size to 5
        x = self.conv_dropout(x)  # Apply dropout after first conv layer
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = self.conv_dropout(x)  # Apply dropout after second conv layer
        x = x.view(-1, 32 * (seq_length // 5 // 2))  # Adjust the division factor according to your new sequence length
        z = z.unsqueeze(1)
        z = F.relu(self.fc1c(z))
        z = self.fc_dropout(z)
        z = F.relu(self.fc2c(z))
        z= self.fc_dropout(z)
        x = torch.cat((x,z),dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)  # Apply dropout before final fully connected layer
        x = self.fc2(x)
        return x
    
class ModifiedSimpleCNNCondOnConv(nn.Module):
    def __init__(self, dropout_rate=0.5, seq_length=100):  # Keep the dropout_rate and seq_length parameters
        super(ModifiedSimpleCNNCondOnConv, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Dropout layer after convolutional layers
        self.conv_dropout = nn.Dropout(p=dropout_rate)

        # Conditional embeddings transformation layers
        self.cond_transform1 = nn.Linear(1, 16)  # Transforming the conditional input to match conv1's feature maps
        self.cond_transform2 = nn.Linear(1, 32)  # Transforming the conditional input to match conv2's feature maps
        
        # Calculate the new sequence length after conv and pooling layers
        new_seq_length = seq_length // 5  # Adjusting for max pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * new_seq_length, 64)  # Adjust the input size according to the output of conv2
        self.fc2 = nn.Linear(64, 1)  # Assuming binary classification
        
        # Dropout layer before the final fully connected layer
        self.fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, z, seq_length=100):
        x = x.permute(0, 2, 1)
        z = z.unsqueeze(1)
        # Transform the conditional input and expand it to match the conv1 output shape
        cond_emb1 = F.relu(self.cond_transform1(z)).unsqueeze(2).expand(-1, -1, x.size(2))
        x = F.max_pool1d(F.relu(self.conv1(x) + cond_emb1), 5)  # Add conditional embedding to conv1 output
        x = self.conv_dropout(x)
        
        # Transform the conditional input and expand it to match the conv2 output shape
        cond_emb2 = F.relu(self.cond_transform2(z)).unsqueeze(2).expand(-1, -1, x.size(2))
        x = F.max_pool1d(F.relu(self.conv2(x) + cond_emb2), 2)  # Add conditional embedding to conv2 output
        x = self.conv_dropout(x)
        
        x = x.view(-1, 32 * (seq_length // 5 // 2))  # Adjust the division factor according to your new sequence length
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)  # Apply dropout before final fully connected layer
        x = self.fc2(x)
        return x

class ModifiedSimpleCNNCondOnConvMultiply(nn.Module):
    def __init__(self, dropout_rate=0.5, seq_length=100):
        super(ModifiedSimpleCNNCondOnConvMultiply, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Dropout layer after convolutional layers
        self.conv_dropout = nn.Dropout(p=dropout_rate)

        # Conditional embeddings transformation layers
        self.cond_transform1 = nn.Linear(1, 16)  # To match conv1's feature maps
        self.cond_transform2 = nn.Linear(1, 32)  # To match conv2's feature maps
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * (seq_length // 5 // 2), 64)
        self.fc2 = nn.Linear(64, 1)  # Assuming binary classification
        
        # Dropout layer before the final fully connected layer
        self.fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, z, seq_length=100):
        x = x.permute(0, 2, 1)
        z = z.unsqueeze(1)

        # Transform the conditional input for conv1
        cond_emb1 = F.relu(self.cond_transform1(z)).unsqueeze(2)
        # Broadcast conditional embedding to match the shape of conv1 output and multiply
        cond_emb1 = cond_emb1.expand(-1, -1, x.size(2))
        x1 = F.relu(self.conv1(x))
        x1 = x1 * cond_emb1  # Element-wise multiplication
        x1 = F.max_pool1d(x1, 5)
        x1 = self.conv_dropout(x1)
        
        # Transform the conditional input for conv2
        cond_emb2 = F.relu(self.cond_transform2(z)).unsqueeze(2)
        # Broadcast conditional embedding to match the shape of conv2 output and multiply
        cond_emb2 = cond_emb2.expand(-1, -1, x1.size(2))
        x2 = F.relu(self.conv2(x1))
        x2 = x2 * cond_emb2  # Element-wise multiplication
        x2 = F.max_pool1d(x2, 2)
        x2 = self.conv_dropout(x2)
        
        # Flatten and pass through fully connected layers
        x2 = x2.view(-1, 32 * (seq_length // 5 // 2))
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc_dropout(x2)
        x2 = self.fc2(x2)
        return x2
