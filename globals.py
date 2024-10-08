#Globals for training

#train datasets
DATA_PATH_TRAIN = 'data\\train_data_imdb_25k.csv'

# develop dataset
DATA_PATH_DEV = 'data\develop_data_imdb_12k.csv'

#test datasets
DATA_PATH_IMDB_TEST = 'data\\test_data_imdb_12k.csv'
DATA_PATH_AMAZON_TEST = 'data\\test_data_amazon.csv'
DATA_PATH_YELP_TEST = 'data\\test_data_yelp.csv'

# model types
MODELS = {"Simple perceptron": 'linear', "Multi linear": 'm_linear', "RNN": 'gv_rnn', "LSTM": 'gv_lstm'}
VOCABS = {"linear": 'vocab_20k', "m_linear": 'vocab_20k', "gv_rnn": 'glo_ve_19k', "gv_lstm": 'glo_ve_19k'}


