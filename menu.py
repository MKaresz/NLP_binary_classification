import train
import predict

import sys
import pandas as pd

from globals import DATA_PATH_AMAZON_TEST, DATA_PATH_YELP_TEST, DATA_PATH_IMDB_TEST, MODELS


MODELS_MENU = {
    1:["Simple perceptron", 'linear'],
    2:["Multi linear", 'm_linear'],
    3:["RNN", 'gv_rnn'],
    4:["LSTM", 'gv_lstm']}

DATASETS = {
    1:["IMDB", DATA_PATH_IMDB_TEST],
    2:["AMAZON", DATA_PATH_AMAZON_TEST],
    3:["YELP", DATA_PATH_YELP_TEST]
    }

def make_evaluation(model_type: str, dataset: int):
    """
    Evaluate the model. Evaluation includes the confusion matrix, 
    the precision-recall curve and the loss on training and validation data 
    during the training of the model.

    Args:
        model: the name of the neural network model
    """
    dataset_choice =(DATASETS.get(dataset))
    df = pd.read_csv(dataset_choice[1])
    predict.get_prediction_from_df(model_type, df)

def make_predict(model_type: str):
    """
    Give a prediction for an input text using the given model. The prediction is
    printed on the terminal.
    """
    text = input("Write a review to get a sentiment prediction from the model:\n")
    predict.get_prediction_from_str(model_type, text)

def make_train(model_type):
    """Train the model with default hyperparameters."""
    train.call_train_torch_model(
        model_type= model_type,
        vocab_name= 'vocab_20K',
        glo_ve_name= 'glo_ve_19k',
        num_epochs= 50,
        batch_size= 32,
        paragraph_size= 100,
        dropout_prob= 0.5,
        embedding_dim= 64,
        hidden_dim= 64,
        output_dim= 1,
        sentiment_threshold= 0.5,
        evaluate_model= True, 
        save_model= False
    )

def main_menu():
    while True:
        print("\nPlease choose a model.")
        print("1:Simple perceptron")
        print("2:Multi linear")
        print("3:RNN")
        print("4:LSTM")
        print("5:Quit")

        choice = input("Select an option: ")
        try:
            choice = int(choice)
        except:
            print("\nInvalid choice. Please choose again.")

        if choice in range(1,5):
            print(f"\nNow using {MODELS_MENU.get(choice)[0]} model.")
            model_submenu(MODELS_MENU.get(choice)[1])
        elif choice == 5:
            sys.exit("Program is exiting.")
        else:
            print("\nInvalid choice. Please choose again.")


def model_submenu(model_type: str):
    while True:
        print("\nPlease choose an option:")
        print("1. Evaluate this model on different datasets.")
        print("2. Make prediction from user input")
        print("3. Train this model on IMDB train dataset.")
        print("4. Back to Main Menu")

        choice = input("Select an option: ")
        try:
            choice = int(choice)
        except:
            print("\nInvalid choice. Please choose again.")

        if choice == 1:
            submenu_dataset(model_type)
        elif choice == 2:
            make_predict(model_type)
        elif choice == 3:
            make_train(model_type)
        elif choice == 4:
            break
        else:
            print("Invalid choice. Please try again.")

def submenu_dataset(model_type: str):
    while True:
        print("\nPlease choose a dataset:")
        print("1. IMDB movie reviews")
        print("2. AMAZON product reviews")
        print("3. YELP restaurant reviews")
        print("4. Back previous Menu")

        choice = input("Select an option: ")
        try:
            choice = int(choice)
        except:
            print("\nInvalid choice. Please choose again.")

        if choice in range(1,4):
            make_evaluation(model_type, choice)
        elif choice == 4:
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()



'''
import file_utils
import model_evaluate
import predict
import train

from collections import OrderedDict
import torch.nn as nn
import pandas as pd

from globals import DATA_PATH_AMAZON_TEST, DATA_PATH_YELP_TEST, DATA_PATH_IMDB_TEST, MODELS

    
def evaluate_fn(model: str):
    """
    Evaluate the model. Evaluation includes the confusion matrix, 
    the precision-recall curve and the loss on training and validation data 
    during the training of the model.

    Args:
        model: the name of the neural network model
    """
    dataset_choice =print_menu(DATASETS)
    if dataset_choice == 4:
        return model
    df = pd.read_csv(DATASETS[dataset_choice][1])
    predict.get_prediction_from_df(MODELS[model], df)

def predict_fn(model):
    """
    Give a prediction for an input text using the given model. The prediction is
    printed on the terminal.
    """
    text = input(f"Write a review to give a prediction for with {model}:\n")
    model = MODELS[model]
    predict.get_prediction_from_str(model, text)

def train_fn(model):
    """Train the model with default hyperparameters."""
    model = MODELS[model]
    train.call_train_torch_model(
        model_type= model,
        vocab_name= 'vocab_20K',
        glo_ve_name= 'glo_ve_19k',
        num_epochs= 50,
        batch_size= 32,
        paragraph_size= 100,
        dropout_prob= 0.5,
        embedding_dim= 64,
        hidden_dim= 64,
        output_dim= 1,
        sentiment_threshold= 0.5,
        evaluate_model= True, 
        save_model= False
    )


def print_menu(menu):
    """Print the menu on the terminal and return the choice."""
    print(25 * "-")
    for key in menu.keys():
        print(key + ":" + menu[key][0])
    choice = input("Make A Choice: ")
    #menu[choice][1]()
    return choice


MENU = OrderedDict([
    ("1" , ("Evaluate this model", evaluate_fn)),
    ("2" , ("Predict", predict_fn)),
    ("3" , ("Train", train_fn)),
    ("4" , ("Quit", exit)),
        ])

MODELS_MENU = OrderedDict([
    ("1", ("Simple perceptron", lambda: print_menu(MENU))),
    ("2", ("Multi linear", lambda: print_menu(MENU))),
    ("3" , ("RNN", lambda: print_menu(MENU))),
    ("4" , ("LSTM", lambda: print_menu(MENU))),
    ("5" , ("Quit", exit)),
        ])

DATASETS = OrderedDict([
    ("1", ("IMDB", DATA_PATH_IMDB_TEST)),
    ("2", ("AMAZON", DATA_PATH_AMAZON_TEST)),
    ("3", ("YELP", DATA_PATH_YELP_TEST)),
    ("4" , ("Quit", exit)),
    ])



if __name__=="__main__":
    print("\nNLP Project, Task 1, binary classification in movie reviews.")
    running = True
    while running:
        choice = print_menu(MODELS_MENU)
        model = MODELS_MENU[choice][0]
        next_nr = MODELS_MENU[choice][1]()
        run_fn = MENU[next_nr][1](model)
'''