import train
import text_utils
import file_utils
import vectorizers
import model_evaluate
from custom_types import ModelParams
from globals import VOCABS, DATA_PATH_IMDB_TEST, DATA_PATH_AMAZON_TEST, DATA_PATH_YELP_TEST

import numpy as np
import pandas as pd
import torch


def get_prediction_from_str(model_type: str, query_text: str):
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device available:", device)

    #load model, model params and vocab for preprocess
    model, model_params = file_utils.load_model(model_type)
    
    cleaned_text = text_utils.clean_text(query_text)
    # load appropriate vocabulary and vectorize input text
    if (model_type == 'gv_lstm') or (model_type == 'gv_rnn'):
        # load vector vocab
        vocab = file_utils.load_glo_ve_vector(VOCABS[model_type])
        # vocab_size
        input_vec = vectorizers.build_glo_ve_vector([cleaned_text], vocab.wv, model_params.paragraph_size)
    else:
        vocab = file_utils.load_vocabulary(VOCABS[model_type])
        # vectorize the input text
        vocab_to_idx = text_utils.build_word_to_idx_vocab(vocab)
        input_vec = vectorizers.build_word_to_idx_vector([cleaned_text], vocab_to_idx, model_params.vocab_size)

    input_tensor = torch.from_numpy(np.array(input_vec))

    model.eval()
    model.to(device)
    with torch.no_grad():
        input_tensor, model = input_tensor.to(device), model.to(device)
        outputs = model(input_tensor).squeeze()
        print("outputs", outputs)
        prediction = "positive" if outputs > model_params.sentiment_threshold else "negative"
        print("input: " + query_text + '"' + '\n', prediction)

def get_prediction_from_df(model_type: str, df_data: pd.DataFrame):
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device available:", device)

    # replace sentiment values to nums for predict
    df_data['sentiment'] = df_data['sentiment'].map({'positive': 1, 'negative': 0})
    # load model
    model, model_params = file_utils.load_model(model_type)
    model.to(device)
    test_loader, _ = train.get_data_loader(model_type=model_type, df=df_data, batch_size=model_params.batch_size,
                                                            paragraph_size=model_params.paragraph_size)
    model_evaluate.evaluate_model(model, test_loader, model_params.sentiment_threshold)
    model_evaluate.plot_training_history(model_params)

