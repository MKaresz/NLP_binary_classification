import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Simple linear model with vocab index embedding
class SimpleLinearClassifier(nn.Module):
    """
    A simple linear classifier for text classification tasks.
    This classifier uses an embedding layer followed by a fully connected layer to perform sentiment analysis or other text classification tasks.

    Parameters:
    vocab_size (int): The size of the vocabulary.
    embedding_dim (int): The dimension of the embeddings.
    paragraph_size (int): The size of the input paragraphs (not directly used in this model but relevant for context).
    output_dim (int): The dimension of the output layer. Default is 1.

    Methods:
    forward(x): Defines the forward pass of the model.
    """
    def __init__(self, vocab_size, embedding_dim, output_dim=1):
        super(SimpleLinearClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average the embeddings across the sequence length
        x = self.fc(x)
        return torch.sigmoid(x) # if self.fc.out_features == 1 else torch.softmax(x, dim=1)

# Initialize Linear the model
def create_linear_model(vocab_size: int, embedding_dim: int, output_dim: int = 1):
    return SimpleLinearClassifier(vocab_size, embedding_dim, output_dim)


# Define the Multi linear model with vocab index embedding
class MultiLinearClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim = 1):
        super(MultiLinearClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average the embeddings across the sequence length
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x) # if self.fc.out_features == 1 else torch.softmax(x, dim=1)

# Initialize Multi Linear model
def create_multi_linear_model(vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int = 1):
    return MultiLinearClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)


# Define the RNN-based model with GloVe embedding
class GloVeRNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_prob, output_dim=1):
        super(GloVeRNNClassifier, self).__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])  # Use the last hidden state
        x = self.fc(x)
        return torch.sigmoid(x) # if self.fc.out_features == 1 else torch.softmax(x, dim=1)

# Initialize GloVe-LSTM model
def create_gv_rnn_model(embedding_dim: int, hidden_dim: int, 
                      dropout_prob: float, output_dim = 1):
    return GloVeRNNClassifier(embedding_dim, hidden_dim, dropout_prob, output_dim)


# Define the LSTM-based model wtih GloVe embedding
class GloVeLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_prob, output_dim=1):
        super(GloVeLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Use the last hidden state
        x = self.fc(x)
        return torch.sigmoid(x) # if self.fc.out_features == 1 else torch.softmax(x, dim=1)

# Initialize GloVe-LSTM model
def create_gv_lstm_model(embedding_dim: int, hidden_dim: int, 
                      dropout_prob: float, output_dim = 1):
    return GloVeLSTMClassifier(embedding_dim, hidden_dim, dropout_prob, output_dim)

