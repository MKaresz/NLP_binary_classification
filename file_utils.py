from custom_types import ModelParams

import torch
import torch.nn as nn
import os
import csv
import sys
import json
import models
from typing import Iterable
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def save_model(model: nn.Module, model_params: ModelParams, model_name: str = 'model_'):
    """
    Saves a PyTorch model and its parameters to a specified directory.

    Args:
    model (nn.Module): The PyTorch model to be saved.
    model_params (ModelParams): The parameters of the model to be saved.
    model_name (str, optional): The base name for the saved model file. Default is 'model_'.

    Returns:
    None

    Example:
    model = MyModel()
    model_params = ModelParams(...)
    save_model(model, model_params, model_name='my_model')
    # This will save the model to 'model/my_model.pth' and its parameters.
    """
    dir_path = 'model'
    filename = 'model_' + model_name + ".pth"
    full_path = os.path.join(dir_path, filename)
    print("SAVING MODEL TO:", full_path)
    torch.save(model.state_dict(), full_path)
    _save_model_params(model_params, model_name)

def load_model(model_type: str) -> tuple:
    """
    from the name of the model the method loads in the model parameter than instantiating
    the model from the parameters and loads into it the saved weights

    Args:
    model_name (str, optional): The base name for the saved model file. Default is 'model_'.

    Returns:
    set(model,model_params) Model and model params
    """
    # load model params for model to be initialized
    model_params = _load_model_params(model_type)
    model = nn.Module

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
        sys.exit("Load model error: Unknown model type, exiting!")

    #construct model name
    dir_path = 'model'
    model_filename = 'model_' + model_type + ".pth"
    full_path = os.path.join(dir_path, model_filename)
    print("LOADING MODEL FROM:", full_path)
    #load weights into instantiated model
    model.load_state_dict(torch.load(full_path, weights_only=True))
    return (model, model_params)

def _save_model_params(model_params: ModelParams, model_name: str):
    """
    Saves the model parameters to a JSON file.

    Args:
    model_params (ModelParams): The parameters of the model to be saved.
    model_name (str): The base name for the saved parameters file.

    Returns:
    None

    Example:
    model_params = ModelParams(...)
    _save_model_params(model_params, 'my_model')
    # This will save the model parameters to 'model/my_model.ini'
    """
    dir_path = 'model'
    filename = 'model_' + model_name + ".ini"
    full_path = os.path.join(dir_path, filename)
    with open(full_path, "w") as f:
        json.dump(model_params.__dict__, f)

def _load_model_params(model_type: str):
    """
    Loads model parameters from a JSON file.

    Args:
    model_name (str): The base name of the file from which to load the parameters.

    Returns:
    ModelParams: An instance of ModelParams containing the loaded parameters.

    Example:
    model_params = _load_model_params('my_model')
    # This will load the model parameters from 'model/my_model.ini'
    """
    dir_path = 'model'
    filename = 'model_' + model_type + ".ini"
    full_path = os.path.join(dir_path, filename)
    with open(full_path) as f:
        data = json.load(f)
        model_params = ModelParams(**data)
        return model_params

def save_vocabulary(vocabulary: Iterable, vocab_name: str = 'model_dictionary'):
    """
    Saves a vocabulary to a JSON file.

    Args:
    vocabulary (Iterable): The vocabulary to be saved.
    vocab_name (str, optional): The base name for the saved vocabulary file. Default is 'model_dictionary'.

    Returns:
    None

    Example:
    vocabulary = ['word1', 'word2', 'word3']
    save_vocabulary(vocabulary, vocab_name='my_vocab')
    # This will save the vocabulary to 'model/my_vocab.dic'
    """
    dir_path = 'model'
    filename = vocab_name + ".dic"
    full_path = os.path.join(dir_path, filename)
    with open(full_path, "w") as f:
        json.dump(list(vocabulary), f)
    print("DICTIONARY SAVED TO: ", full_path)

def load_vocabulary(model_name: str):
    """
    Loads a vocabulary from a JSON file.

    Args:
    model_name (str, optional): The base name of the file from which to load the vocabulary. Default is 'vocab_12K'.

    Returns:
    list: The loaded vocabulary as a list of words.

    Example:
    vocabulary = load_vocabulary('my_vocab')
    # This will load the vocabulary from 'model/my_vocab.dic'
    """
    dir_path = 'model'
    filename = model_name + ".dic"
    full_path = os.path.join(dir_path, filename)
    with open(full_path) as f:
        print("DICTIONARY IS LOADED FROM: ", full_path)
        return json.load(f)
    
def split_data_to_three(data: np.array, train_size = 0.7, develop_size = 0.15, save_to_disk = False,
                        export_name = "Data"):
    """
    Splits the data into training, development, and test sets.

    Args:
    data (np.array): The input data to be split.
    train_size (float, optional): The proportion of the data to be used for training. Default is 0.7.
    develop_size (float, optional): The proportion of the data to be used for development. Default is 0.15.
    save_to_disk (bool, optional): If True, saves the splits to CSV files. Default is False.
    export_name (str, optional): The base name for the saved CSV files. Default is "Data".

    Returns:
    tuple: A tuple containing the training, development, and test sets as pandas DataFrames.

    Example:
    data = np.array([...])
    train, develop, test = split_data_to_three(data, train_size=0.6, develop_size=0.2, save_to_disk=True, export_name='my_data')
    # This will split the data and save the splits to 'data/train_my_data.csv', 'data/develop_my_data.csv', and 'data/test_my_data.csv'
    """
    train_percent = int((len(data)*train_size))
    develop_percent = int((len(data)*train_size)+(len(data)*develop_size))
    # cut up df and Resets the index of the DataFrame to start from index 0!!!!
    train = data[:train_percent].reset_index(drop=True)
    develop = data[train_percent:develop_percent].reset_index(drop=True)
    test = data[develop_percent:].reset_index(drop=True)
    if save_to_disk:
        folder_path = "data"
        train.to_csv(os.path.join(folder_path, "train_"+export_name+".csv"), sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)
        develop.to_csv(os.path.join(folder_path, "develop_"+export_name+".csv"), sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)
        test.to_csv(os.path.join(folder_path, "test_"+export_name+".csv"), sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)
    return (train, develop, test)

def split_data_to_two(data: np.array, split_size = 0.5, save_to_disk = False, export_name = "Data"):
    """
    Splits the data into development and test sets.

    Args:
    data (np.array): The input data to be split.
    split_size (float, optional): The proportion of the data to be used for the development set. Default is 0.5.
    save_to_disk (bool, optional): If True, saves the splits to CSV files. Default is False.
    export_name (str, optional): The base name for the saved CSV files. Default is "Data".

    Returns:
    tuple: A tuple containing the development and test sets as pandas DataFrames.

    Example:
    data = np.array([...])
    develop, test = split_data_to_two(data, split_size=0.6, save_to_disk=True, export_name='my_data')
    # This will split the data and save the splits to 'data/develop_my_data.csv' and 'data/test_my_data.csv'
    """
    split_percent = int((len(data)*split_size))
    # cut up df and Resets the index of the DataFrame to start from index 0!!!!
    develop = data[:split_percent].reset_index(drop=True)
    test = data[split_percent:].reset_index(drop=True)
    if save_to_disk:
        folder_path = "data"
        develop.to_csv(os.path.join(folder_path, "develop_"+export_name+".csv"), sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)
        test.to_csv(os.path.join(folder_path, "test_"+export_name+".csv"), sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)
    return (develop, test)

def save_glo_ve_model(model, model_name: str):
    """
    Saves a GloVe model to a specified directory.

    Args:
    model: The GloVe model to be saved.
    model_name (str): The base name for the saved model file.

    Returns:
    None

    Example:
    model = build_glo_ve_model('reviews.csv')
    save_glo_ve_model(model, 'my_glove_model')
    # This will save the model to 'model/my_glove_model.model'
    """
    dir_path = 'model'
    filename = model_name + ".model"
    full_path = os.path.join(dir_path, filename)
    model.save(full_path)

def load_glo_ve_model(model_name: str):
    """
    Loads a GloVe model from a specified directory.

    Args:
    model_name (str): The base name of the file from which to load the model.

    Returns:
    Word2Vec: The loaded GloVe model.

    Example:
    model = load_glo_ve_model('my_glove_model')
    # This will load the model from 'model/my_glove_model.model'
    """
    dir_path = 'model'
    filename = model_name + ".model"
    full_path = os.path.join(dir_path, filename)
    return Word2Vec.load(full_path)

def save_glo_ve_vector(gv_vector, model_name: str):
    """
    Saves a GloVe vector to a specified directory.

    Args:
    gv_vector: The GloVe vector to be saved.
    model_name (str): The base name for the saved vector file.

    Returns:
    None

    Example:
    gv_vector = some_glove_vector
    save_glo_ve_vector(gv_vector, 'my_glove_vector')
    # This will save the vector to 'model/my_glove_vector.gv'
    """
    dir_path = 'model'
    filename = model_name + ".gv"
    full_path = os.path.join(dir_path, filename)
    gv_vector.save(full_path)

def load_glo_ve_vector(model_name: str):
    """
    Loads a GloVe vector from a specified directory.

    Args:
    model_name (str): The base name of the file from which to load the vector.

    Returns:
    KeyedVectors: The loaded GloVe vector.

    Example:
    gv_vector = load_glo_ve_vector('my_glove_vector')
    # This will load the vector from 'model/my_glove_vector.gv'
    """
    dir_path = 'model'
    filename = model_name + ".gv"
    full_path = os.path.join(dir_path, filename)
    return KeyedVectors.load(full_path, mmap='r')


###################
### cut up data 
# cut up data and save to disk
#df_train, df_develop, df_test = file_utils.split_data_to_three(df, save_to_disk=True, export_name="data_imdb")
#df_develop, df_test = file_utils.split_data_to_two(df_test, save_to_disk=True, export_name="data_imdb")

#my_model = load_model('linear')
#print(my_model)