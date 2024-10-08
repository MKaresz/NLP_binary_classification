import globals
import models
import file_utils
import text_utils
import vectorizers
import model_evaluate
from custom_types import ModelParams
from custom_types import ReviewLabelDataset
from custom_types import GloVeDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from gensim import utils
import sys
from datetime import datetime
import pandas as pd
import time


def validate_epoch(model: nn.Module, last_best_model: nn.Module, model_params: ModelParams, develop_loader: DataLoader, 
                   criterion: nn.BCELoss, min_validation_loss: float, save_model: bool = True):
    """
    Validates the current model for one epoch and saves the model if the validation loss is lower than the minimum validation loss.

    Args:
    model (nn.Module) : The model to be validated.
    last_best_model (nn.Module): The model with the best validation loss from previous epochs.
    model_params (ModelParams): The parameters of the model.
    develop_loader (DataLoader): The DataLoader for the validation dataset.
    criterion (nn.BCELoss): The loss function used for validation.
    min_validation_loss (float): The minimum validation loss recorded so far.
    save_model (bool):  Whether to save the model if it achieves a new minimum validation loss. Default is True.

    Returns:
    tuple: A tuple containing the updated minimum validation loss, the normalized validation loss for 
    the current epoch, and the model with the best validation loss.
    """
    model.eval()
    validation_loss = 0
    last_best_model = last_best_model
    for data, labels in develop_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        outputs = model(data)
        loss = criterion(outputs.squeeze(), labels.float())
        validation_loss += loss.item() * data.size(0)
    
    normalized_validation_loss = validation_loss/len(develop_loader)
    if (min_validation_loss > normalized_validation_loss):
        print(f"min_validation_loss: {min_validation_loss} > normalized_validation_loss: {normalized_validation_loss}")
        min_validation_loss = normalized_validation_loss
        last_best_model = model
        if save_model:
            print(f"Saving model with new validation loss: {normalized_validation_loss:.4f}")
            file_utils.save_model(model, model_params, 'validation_' + model_params.model_type)
    return (min_validation_loss, normalized_validation_loss, last_best_model)

def train_model(model_type: str, num_epochs: int, train_loader: DataLoader, develop_loader: DataLoader, 
                model_params: ModelParams, sentiment_threshold: float, save_model: bool = False, evaluate: bool = True):
    """
    Supervised model training on input_data data set using input_label

    Args:
    model_type (str): type of model that is trained.
    num_epochs (int): number of epoch in training.
    train_loader (DataLoader): data handle for training.
    develop_loader (DataLoader): data handle for evaluation.
    model_params (ModelParams): model params from saving model.
    sentiment_threshold (float): The threshold for sentiment analysis.
    evaluate_model (bool): Whether to evaluate the model after training.
    save_model (bool): Whether to save the model after training.

    Returns: 
    None
    """

    #start timer to measure training time with necessary additions
    start_time = time.perf_counter()
    print("Model training started...")

    
    # create model based on model_params
    if (model_params.model_type == 'linear'):
        print("Creating Linear Model")
        model = models.create_linear_model(vocab_size = model_params.vocab_size, 
                                           embedding_dim = model_params.embedding_dim,
                                           output_dim = model_params.output_dim
                                           )
    elif (model_params.model_type == 'm_linear'):
        print("Creating Multi-Linear Model")
        model = models.create_multi_linear_model(vocab_size = model_params.vocab_size, 
                                                 embedding_dim = model_params.embedding_dim,
                                                 hidden_dim = model_params.hidden_dim,
                                                 output_dim = model_params.output_dim
                                                 )
    elif (model_params.model_type == 'gv_rnn'):
        print("Creating RNN Model")
        model = models.create_gv_rnn_model(embedding_dim = model_params.embedding_dim,
                                        hidden_dim = model_params.hidden_dim,
                                        dropout_prob = model_params.dropout_prob,
                                        output_dim = model_params.output_dim
                                        )
    elif (model_params.model_type == 'gv_lstm'):
        print("Creating GloVe LSTM Model")
        model = models.create_gv_lstm_model(embedding_dim = model_params.embedding_dim,
                                         hidden_dim = model_params.hidden_dim,
                                         dropout_prob = model_params.dropout_prob,
                                         output_dim = model_params.output_dim
                                         )
    else:
        sys.exit("Train error: Unknown model type, exiting!")
    
    # printing out model attributes
    print(model)
    print(model_params)

    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device available:",device)
    # put on available device
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # set params for evaluation
    min_validation_loss = np.Infinity
    last_best_model = None
    model_params.train_history = {'training_loss': [], 'validation_loss': []}

    for epoch in range(num_epochs):
        training_loss = 0
        model.train()
        for reviews, labels in train_loader:
            # Gradient clipping for handling gradient exploding
            #for param in model.parameters():
            #    torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
            reviews, labels = reviews.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(reviews)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            training_loss += loss.item() * reviews.size(0) # batch size
            optimizer.step()
        epoch_loss = training_loss / len(train_loader)
        # store new low in validation loss
        min_validation_loss, normalized_validation_loss, last_best_model = validate_epoch(model, last_best_model, model_params, 
                                                              develop_loader, criterion, min_validation_loss,
                                                              save_model)
        # store loss values for plot
        model_params.train_history['training_loss'].append(epoch_loss)
        model_params.train_history['validation_loss'].append(normalized_validation_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # measure elapsed time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.0f} seconds")

    if evaluate:
        model_evaluate.evaluate_model(last_best_model, develop_loader, sentiment_threshold)
        model_evaluate.plot_training_history(model_params)


def call_train_torch_model(model_type: str, vocab_name: str, glo_ve_name: str, num_epochs: int, batch_size: int,
                       paragraph_size: int, dropout_prob: float, embedding_dim: int, hidden_dim: int, output_dim: int, 
                       sentiment_threshold: float, evaluate_model: bool, save_model: bool):
    """
    Loads the vocabulary and performs text vectorization, then calls the training method for the specified 
    model type.

    Parameters:
    model_type (str): The type of model to train. Can be 'linear', 'm_linear', 'rnn', or 'lstm'.
    vocab_name (str): The name of the file containing the vocabulary.
    glo_ve_name (str): The name of the file containing the GloVe vectors.
    num_epochs (int): The number of epochs to train the model.
    batch_size (int): The size of the batches for training.
    paragraph_size (int): The size of the paragraphs for input data.
    dropout_prob (float): The dropout probability for the model.
    embedding_dim (int): The dimension of the embeddings.
    hidden_dim (int): The dimension of the hidden layers.
    output_dim (int): The dimension of the output layer.
    sentiment_threshold (float): The threshold for sentiment analysis.
    evaluate_model (bool): Whether to evaluate the model after training.
    save_model (bool): Whether to save the model after training.

    Returns:
    None
    """

    # define seed to get the same random numbers every time for consistency and reproducibility
    torch.manual_seed(13)

    print("Loading training data...")

    # Load the dataset
    df_train = pd.read_csv(globals.DATA_PATH_TRAIN)
    df_develop = pd.read_csv(globals.DATA_PATH_DEV)
    # Shuffle the DataFrame rows only for train
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    
    # for labels replace string to 0-1 values for vectorization
    df_train['sentiment'] = df_train['sentiment'].map({'positive': 1, 'negative': 0})
    df_develop['sentiment'] = df_develop['sentiment'].map({'positive': 1, 'negative': 0})

    # Shuffle the DataFrame rows only for train
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    train_loader, vocab = get_data_loader(model_type=model_type, df=df_train, batch_size=batch_size,
                                                          paragraph_size=paragraph_size)
    develop_loader, _ = get_data_loader(model_type=model_type, df=df_develop, batch_size=batch_size,
                                                          paragraph_size=paragraph_size)

    if (model_type == 'gv_lstm') or (model_type == 'gv_rnn'):
        vocab_size = len(vocab.wv.key_to_index)
    else:
        vocab_size = len(vocab)

    # change embedding dim to pass embedding of GloVe
    if (model_type == 'gv_lstm') or (model_type == 'gv_rnn'):
        # this needs to pass to the generated vector size coming from the KeyedVector matrix
        embedding_dim=vocab.vector_size

    # create model params
    model_params = ModelParams(model_type=model_type, vocab_size=vocab_size, batch_size=batch_size,
                            paragraph_size=paragraph_size, dropout_prob=dropout_prob, 
                            embedding_dim=embedding_dim, hidden_dim=hidden_dim, 
                            output_dim=output_dim, sentiment_threshold=sentiment_threshold,
                            train_history = None)

    #train model
    train_model(model_type=model_type, num_epochs=num_epochs, train_loader=train_loader, develop_loader=develop_loader, 
                    model_params=model_params, sentiment_threshold=sentiment_threshold, save_model=save_model, evaluate=evaluate_model)

def get_data_loader(model_type: str, df: pd.DataFrame, batch_size: int, paragraph_size: int):
    # clean develop corpora for testing
    cleaned_corpus = text_utils.clean_corpus(corpus=df['review'].tolist())

    if (model_type == 'gv_lstm') or (model_type == 'gv_rnn'):
        # load vector vocab
        vocab = file_utils.load_glo_ve_vector(globals.VOCABS[model_type])
        X_vec = np.array(vectorizers.build_glo_ve_vector(cleaned_corpus, vocab.wv, paragraph_size))
        y_vec = df['sentiment'].tolist()
        
        dataset = GloVeDataset(X_vec, y_vec)
    else:
        # load vocab
        vocab = file_utils.load_vocabulary(globals.VOCABS[model_type])
        # convert words to indexes in vocab
        vocab_to_idx = text_utils.build_word_to_idx_vocab(vocab)
        # create sized vectors using word indexes => for padding using 1 and un-knows using 0
        X_vec = vectorizers.build_word_to_idx_vector(cleaned_corpus, vocab_to_idx, paragraph_size)
        y_vec = df['sentiment'].tolist()

        dataset = ReviewLabelDataset(X_vec, y_vec)

    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
    return (data_loader, vocab)

def make_vocabulary( vocab_size: int = 5000, file_name: str = 'model_dictionary',
                    is_save_enabled: bool = False):
    print("Building vocabulary...")
    df_train = pd.read_csv(globals.DATA_PATH_TRAIN)
    cleaned_train_corpora = text_utils.clean_corpora(corpora=df_train['review'].tolist())
    vocab = text_utils.build_vocab(cleaned_train_corpora, vocab_size=vocab_size, file_name=file_name,
                    is_save_enabled=is_save_enabled)
    if is_save_enabled:
        print(f"Vocabulary is generated, with {len(vocab)} num. of words, saved as {file_name}")
    else:
        print(f"Vocabulary is generated, with {len(vocab)} num. of words.")




###################################################
## testing train
'''
# Accuracy: 74.33%
call_train_torch_model(
    model_type= 'gv_rnn', # 'linear',  'm_linear',  'rnn',  'lstm', 'gv_lstm'
    vocab_name= 'vocab_1K', # 'vocab_100', 'vocab_1K', 'vocab_25K', 'vocab_50K' => recommended 'vocab_25K' 
    glo_ve_name= 'glo_ve_25k', #'glo_ve_1k', 'glo_ve_25k', 'glo_ve_50k'
    num_epochs= 20, # 300
    batch_size= 16, # 64, 128 ,256, 512
    paragraph_size= 100, # 500 - 700
    dropout_prob= 0.25,
    embedding_dim= 32, # 100 - 300
    hidden_dim= 32, #32-256
    output_dim= 1,
    sentiment_threshold= 0.5, # from precision-recall graph!
    evaluate_model= True,
    save_model= True
    )
'''

'''
# 71.02%
call_train_torch_model(
    model_type= 'gv_rnn', # 'linear',  'm_linear',  'rnn',  'lstm', 'gv_lstm'
    vocab_name= 'vocab_1K', # 'vocab_100', 'vocab_1K', 'vocab_25K', 'vocab_50K' => recommended 'vocab_25K' 
    glo_ve_name= 'glo_ve_25k', #'glo_ve_1k', 'glo_ve_25k', 'glo_ve_50k'
    num_epochs= 20, # 300
    batch_size= 16, # 64, 128 ,256, 512
    paragraph_size= 200, # 500 - 700
    dropout_prob= 0.25,
    embedding_dim= 32, # 100 - 300
    hidden_dim= 32, #32-256
    output_dim= 1,
    sentiment_threshold= 0.5, # from precision-recall graph!
    evaluate_model= True,
    save_model= True
    )
'''

# 64.03%
'''
call_train_torch_model(
    model_type= 'gv_rnn', # 'linear',  'm_linear',  'rnn',  'lstm', 'gv_lstm'
    vocab_name= 'vocab_1K', # 'vocab_100', 'vocab_1K', 'vocab_25K', 'vocab_50K' => recommended 'vocab_25K' 
    glo_ve_name= 'glo_ve_25k', #'glo_ve_1k', 'glo_ve_25k', 'glo_ve_50k'
    num_epochs= 20, # 300
    batch_size= 16, # 64, 128 ,256, 512
    paragraph_size= 100, # 500 - 700
    dropout_prob= 0.25,
    embedding_dim= 64, # 100 - 300
    hidden_dim= 64, #32-256
    output_dim= 1,
    sentiment_threshold= 0.5, # from precision-recall graph!
    evaluate_model= True,
    save_model= True
    )
'''

'''
# Accuracy: 87.74%
call_train_torch_model(
    model_type= 'gv_lstm', # 'linear',  'm_linear',  'rnn',  'lstm', 'gv_lstm'
    vocab_name= 'vocab_1K', # 'vocab_100', 'vocab_1K', 'vocab_25K', 'vocab_50K' => recommended 'vocab_25K' 
    glo_ve_name= 'glo_ve_25k', #'glo_ve_1k', 'glo_ve_25k', 'glo_ve_50k'
    num_epochs= 20, # 300
    batch_size= 32, # 64, 128 ,256, 512
    paragraph_size= 200, # 500 - 700
    dropout_prob= 0.25,
    embedding_dim= 256, # 100 - 300
    hidden_dim= 512, #32-256
    output_dim= 1,
    sentiment_threshold= 0.5, # from precision-recall graph!
    evaluate_model= True,
    save_model= True
    )
'''

'''
# GV LSTM 83.3%
call_train_torch_model(
    model_type= 'gv_lstm', # 'linear',  'm_linear',  'rnn',  'lstm', 'gv_lstm'
    vocab_name= 'vocab_1K', # 'vocab_100', 'vocab_1K', 'vocab_25K', 'vocab_50K' => recommended 'vocab_25K' 
    glo_ve_name= 'glo_ve_25k', #'glo_ve_1k', 'glo_ve_25k', 'glo_ve_50k'
    num_epochs= 20, # 300
    batch_size= 32, # 64, 128 ,256, 512
    paragraph_size= 100, # 500 - 700
    dropout_prob= 0.25,
    embedding_dim= 64, # 100 - 300
    hidden_dim= 128, #32-256
    output_dim= 1,
    sentiment_threshold= 0.5, # from precision-recall graph!
    evaluate_model= True,
    save_model= True
    )
'''


'''
#RNN 70%
call_train_torch_model(
    model_type= 'gv_rnn', # 'linear',  'm_linear',  'rnn',  'lstm', 'gv_lstm'
    vocab_name= 'vocab_1K', # 'vocab_100', 'vocab_1K', 'vocab_25K', 'vocab_50K' => recommended 'vocab_25K' 
    glo_ve_name= 'glo_ve_25k', #'glo_ve_1k', 'glo_ve_25k', 'glo_ve_50k'
    num_epochs= 100, # 300
    batch_size= 32, # 64, 128 ,256, 512
    paragraph_size= 100, # 500 - 700
    dropout_prob= 0.25,
    embedding_dim= 32, # 100 - 300
    hidden_dim= 32, #32-256
    output_dim= 1,
    sentiment_threshold= 0.5, # from precision-recall graph!
    evaluate_model= True,
    save_model= True
    )
'''

'''
#LINEAR MODELS GOOOOOOOD!
model_type= 'gv_rnn' # 'linear',  'm_linear', 'gv_lstm', 'gv_rnn'
call_train_torch_model(
    model_type= model_type,
    vocab_name= 'vocab_20k',

    num_epochs= 100, # 300
    batch_size= 32, # 64, 128 ,256, 512
    paragraph_size= 100, # 500 - 700

    glo_ve_name= 'glo_ve_19k',
    dropout_prob= 0.25,
    embedding_dim= 64, # 100 - 300
    hidden_dim= 64, #32-256
    output_dim= 1,

    sentiment_threshold= 0.8, # from precision-recall graph!
    
    evaluate_model= True,
    save_model= True
    )
'''


'''
# RNN 95% on 1K database!?!?! but 50% on 25K database
call_train_torch_model(
    model_type= 'gv_rnn', # 'linear',  'm_linear',  'rnn',  'lstm', 'gv_lstm'
    vocab_name= 'vocab_1K', # 'vocab_100', 'vocab_1K', 'vocab_25K', 'vocab_50K' => recommended 'vocab_25K' 
    glo_ve_name= 'glo_ve_25k', #'glo_ve_1k', 'glo_ve_25k', 'glo_ve_50k'
    num_epochs= 100, # 300
    batch_size= 32, # 64, 128 ,256, 512
    paragraph_size= 100, # 500 - 700
    dropout_prob= 0.25,
    embedding_dim= 64, # 100 - 300
    hidden_dim= 64, #32-256
    output_dim= 1,
    sentiment_threshold= 0.5, # from precision-recall graph!
    evaluate_model= True,
    save_model= True
    )
'''

'''
call_train_torch_model(
    model_type= 'gv_lstm', # 'linear',  'm_linear', 'gv_lstm', 'gv_rnn'
    vocab_name= 'vocab_20k',

    num_epochs= 50, # 300
    batch_size= 32, # 64, 128 ,256, 512
    paragraph_size= 200, # 500 - 700

    glo_ve_name= 'glo_ve_19k',
    dropout_prob= 0.25,
    embedding_dim= 64, # 100 - 300
    hidden_dim= 64, #32-256
    output_dim= 1,

    sentiment_threshold= 0.5, # from precision-recall graph!
    
    evaluate_model= True,
    save_model= True
    )
'''
