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

#Globals for Evaluate
SENTIMENT_THRESHOLD = 0.5 # movie scores above 7 or more stars are => Good


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



#########################################
### DEBUG PREDICTOR
'''
text_p_1 = "An absolute delight from start to finish, this film captivates with its heartwarming story and stellar performances. A must-watch that leaves you feeling uplifted and inspired!"
text_p_2 = "This movie is a joyous celebration of life, filled with unforgettable characters and a touching narrative. It’s a cinematic gem that will leave you smiling long after the credits roll!"
text_p_3 = "A beautifully crafted film that enchants with its compelling storyline and breathtaking visuals. This cinematic masterpiece is sure to leave a lasting impression on your heart."
text_n_1 = "Despite its promising premise, the film falls flat with a lackluster plot and uninspired performances. It’s a disappointing watch that fails to engage or entertain."
text_n_2 = "This movie is a tedious experience, marred by poor pacing and unconvincing acting. It’s a forgettable film that leaves much to be desired."
text_n_3 = "An overhyped film that doesn’t live up to its expectations, plagued by a weak script and clichéd characters. It’s a frustrating watch that ultimately feels like a waste of time."

print("---------------- POSITIVE ------------------")
get_prediction_from_str(model_type='linear',query_text=text_p_1)
get_prediction_from_str(model_type='m_linear',query_text=text_p_2)
get_prediction_from_str(model_type='gv_rnn',query_text=text_p_3)
print("---------------- NEGATIVE ------------------")
get_prediction_from_str(model_type='gv_lstm',query_text=text_n_1)
get_prediction_from_str(model_type='linear',query_text=text_n_2)
get_prediction_from_str(model_type='m_linear',query_text=text_n_3)
'''


#####################################
### EVAL CORPORA LOADED MODEL LOADED CORPORA
# test method
'''
model_type= 'linear' # 'linear',  'm_linear', 'gv_lstm', 'gv_rnn'
df_yelp = pd.read_csv(globals.DATA_PATH_YELP_TEST)
get_prediction_from_df(model_type, df_data=df_yelp)

df_amazon = pd.read_csv(globals.DATA_PATH_YELP_TEST)
get_prediction_from_df(model_type, df_data=df_amazon)

df_imdb = pd.read_csv(globals.DATA_PATH_YELP_TEST)
get_prediction_from_df(model_type, df_data=df_imdb)
'''


'''

df_test = pd.read_csv(DATA_PATH_YELP_TEST)
# replace sentiment to ints
df_test['sentiment'] = df_test['sentiment'].map({'positive': 1, 'negative': 0})

model_type = "linear"
model, model_params = file_utils.load_model(model_type)
emb_dim = model.embedding.embedding_dim
test_loader, _ = train.get_data_loader(model_type=model_type, df=df_test, batch_size=model_params.batch_size,
                                                          paragraph_size=model_params.paragraph_size)
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)
model_evaluate.evaluate_model(model, test_loader, model_params.sentiment_threshold)
'''


'''
model.eval()
with torch.no_grad():
    for reviews, labels in test_loader:
        reviews, labels = reviews.to("cpu"), labels.to("cpu")
        outputs = model(reviews).squeeze()
        #print("outputs", outputs)
        prediction = "positive" if outputs > model_params.sentiment_threshold else "negative"
        #print("input: " + reviews + '"' + '\n', prediction)
'''

'''
with torch.no_grad():
    input_tensor, model = input_tensor.to(device), model.to(device)
    outputs = model(input_tensor).squeeze()
    print("outputs", outputs)
    prediction = "positive" if outputs > model_params.sentiment_threshold else "negative"
    print("input: " + query_text + '"' + '\n', prediction)
'''
#model_evaluate.evaluate_model(model, test_loader, model_params.sentiment_threshold)
#model_evaluate.plot_training_history(model_params)



# from custom_types import ReviewLabelDataset
# import pandas as pd
# from train import DATA_PATH_DEV, PARAGRAPH_SIZE, VOCAB_NAME, BATCH_SIZE
# from torch.utils.data import DataLoader

# vocab = file_utils.load_vocabulary(VOCAB_NAME)
# vocab_size = len(vocab)
# vocab_to_idx = text_utils.build_word_to_idx_vocab(vocab)
   
# df_develop = pd.read_csv(DATA_PATH_DEV)
# df_develop['sentiment'] = df_develop['sentiment'].map({'positive': 1, 'negative': 0})

# cleaned_corpora = text_utils.clean_corpora(corpora=df_develop['review'].tolist())
# X_develop_vec = vectorizers.build_word_to_idx_vector(cleaned_corpora, vocab_to_idx, PARAGRAPH_SIZE)

# y_vec = df_develop['sentiment'].tolist()
# develop_dataset = ReviewLabelDataset(X_develop_vec, y_vec)
# develop_loader = DataLoader(develop_dataset, batch_size=BATCH_SIZE, shuffle=False)


# for i, text in enumerate(cleaned_corpora[:15]):
#     get_torch_prediction(model_name="m_linear", query_text=text)
#     print("REAL sentiment: ", y_vec[i])
    
# print("EVALUATING")
# from evaluate import evaluate_model

# model, model_params = get_model("m_linear")
# evaluate_model(model, develop_loader, "cpu")





























'''


    # clean text
    #TODO dim_size for text -> how many words are vectorized in one block?!
    #TODO where to put dim_size it, it's sohuld be as a new param with the model ?!
    #BUG FIX THIS MAGIC NUMBER!!!!
    vectorized_text = text_utils.text_to_tf_idf_vec(query_text,10)
    # print(vectorized_text, type(vectorized_text))
    
    # feed it to the model
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device available:",device)
    
    # Convert NumPy array to PyTorch tensor
    sentence_vector = torch.tensor(vectorized_text, dtype=torch.float32).to(device)
    print(sentence_vector, sentence_vector.dtype)
    # Make the prediction

    #TODO WTF? WHY IS IT NOT CALLABLE!!!

    ########################################
    #COMPARE MODELS!!!!
    ########################################

    model.eval()
    with torch.no_grad():
        outputs = model(sentence_vector).squeeze()
        predicted = (outputs > 0.5).float()
        # return prediction
        prediction = "positive" if predicted >0.5 else "negative"
        print(predicted, prediction)
    
    '''
'''

# from AI >> Make the prediction
#with torch.no_grad():
#    outputs = model([**inputs)
#    logits = outputs.logits

# Convert logits to probabilities (if needed)
#probabilities = torch.softmax(logits, dim=1)
'''
    

#########################################
### DEBUG GET MODEL
'''
#1 get choosen model
model = file_utils.load_model("model_linear")
#2 get dictionary
vocab_ord = file_utils.load_dictionary_ordered('vocab_12K')
#3 get input text
input_text = "The master of movie spectacle Cecil B. De Mille goes West. Using three legends of the old west as its protagonists (they probably never met),Gary Cooper is portraying Wild Bill Hickock,James Ellison as Buffalo Bill and Jean Arthur does make a nice Calamity Jane. The story serves only for De Mille to hang some marvelous action sequences on, like the big Indian attack.Scenes like that are extremely well done.If you don't mind the somewhat over-the-top performances of the cast this is an very entertaining western.Look out for a very young Anthony Quinn essaying the role of an Indian brave who participated at the battle of Little Big Horn.This part got him at least noticed in Hollywood."
#4 get clean input text
cleaned_text = text_utils.clean_text(input_text)
#5 vectorize input text
# padding to fixed size convert to vectors
X_train_vec = text_utils.preprocess_corpora_tf(df_train['review'].to_numpy(), VOCAB_NAME, True)

#5 give vector to model for prediction

model.eval()
    with torch.no_grad():
        output = model(input_sequence.unsqueeze(0))

'''



'''
# Post-process the output (e.g., take the argmax)
predicted_next_number = torch.argmax(output, dim=1).item()
print(f'Predicted next number: {predicted_next_number:.4f}')
'''































