import train
import predict
import model_evaluate
import file_utils

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
    _, model_params = file_utils.load_model(model_type=model_type)
    train.call_train_torch_model(
        model_type=model_type,
        vocab_name='vocab_1K',
        glo_ve_name='glo_ve_25k',
        num_epochs=20,
        batch_size=model_params.batch_size,
        paragraph_size=model_params.paragraph_size,
        dropout_prob=model_params.dropout_prob,
        embedding_dim=model_params.embedding_dim,
        hidden_dim=model_params.hidden_dim,
        output_dim=model_params.output_dim,
        sentiment_threshold=model_params.sentiment_threshold,
        evaluate_model= True,
        save_model= True
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
