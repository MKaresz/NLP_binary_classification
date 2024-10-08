import torch
from torch.utils.data import Dataset

# custom model parameters class to store model data
class ModelParams():
    """
    A class to represent the parameters for different types of models in pytorch.

    It's used for the following model types: 'linear', 'm_linear', 'gv_rnn', 'gv_lstm' and 
    can be saved with the model into a jsons file.
    
    Args: 
        model_type: 'linear', 'm_linear', 'gv_rnn', 'gv_lstm'
        vocab_size: the size of the vocabulary used with the model
        paragraph_size: the size of one paragraph used with the model
        dropout_prob: dropout rate in float 0.0 - 1.0
        embedding_dim: embedding dim used for gv_rnn, gv_lstm
        hidden_dim: hidden dimension used with the model
        output_dim: output dimension for binary classification is 1 ( 0 or 1 )
        sentiment_threshold: value to divide probability between false or true ( 0.0 - 1.0 )
        train_history: stored training loss, and validation loss for plotting
    """
    def __init__(self, model_type='linear', vocab_size=15000, batch_size=32, paragraph_size=300, dropout_prob=0.5, 
                 embedding_dim=5, hidden_dim=10, output_dim=1, sentiment_threshold=0.5,
                 train_history: dict = {'training_loss': [], 'validation_loss': []}):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.paragraph_size = paragraph_size
        self.dropout_prob = dropout_prob
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sentiment_threshold = sentiment_threshold
        self.train_history = train_history

    def __str__(self):
        return f"model_type: {self.model_type}, vocab_size:{self.vocab_size}, batch_size:{self.batch_size}, paragraph_size: {self.paragraph_size}, dropout_prob:{self.dropout_prob}, embedding_dim:{self.embedding_dim}, hidden_dim:{self.hidden_dim}, output_dim:{self.output_dim}, sentiment_threshold:{self.sentiment_threshold}"

# custom dataset class to handle the data
class ReviewLabelDataset(Dataset):
    """
    Data class for linear and multi linear models.
    Custom dataset class to handle the data.
    """
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = torch.tensor(self.reviews[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return review, label


class GloVeDataset(Dataset):
    """
    Data class for GloVe RNN and GloVe LSTM models.
    Custom dataset class to handle the GloVe embedding.
    """
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = torch.tensor(self.reviews[idx], dtype=torch.float32).clone().detach().requires_grad_(True)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return review, label

