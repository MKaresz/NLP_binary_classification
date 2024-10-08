import file_utils
import vectorizers

import sys
import re
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import Iterable
from gensim import utils
from gensim.models import Word2Vec


def _chr_filter(word: str) -> str:
    """
    Converts a word to lowercase and removes all non-alphabetic characters.

    Args:
    word (str): The input word to be filtered.

    Returns:
    str: The filtered word in lowercase with non-alphabetic characters removed.

    Example:
    word = "Hello, World123!"
    result = _chr_filter(word)  # result will be "helloworld"
    """
    #  casefold() makes lowercase chars in a way that is more aggressive than lower()
    word = word.casefold()
    # remove all non-word characters
    word = re.sub(r"[\W\d]+",'',word)
    return word

def _lemmatize_with_pos(clean_text: list[str]) -> list[str]:
    """
    Lemmatizes the input text using part-of-speech (POS) tagging to determine the correct lemma for each token.

    Args:
    clean_text (str): The input text to be lemmatized.

    Returns:
    list[str]: A list of lemmatized tokens.

    Example:
    clean_text = ['the', 'striped', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'good']
    print(_lemmatize_with_pos(clean_text))
    ['the', 'striped', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'good']
    """
    cleaned_tokens = []
    for token, tag in pos_tag(clean_text):
        # n for noun, v for verb, a for adjective , r for adverb.
        if tag.startswith('N'):
            pos = 'n'
        elif tag.startswith('V'):
            pos = 'v'
        else:
            pos = 'a'

        # return the base  form of a word (lemma)
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize(token, pos)
        cleaned_tokens.append(lemma)
    return cleaned_tokens

def clean_corpus(corpus: list[str], count_average_num_word: bool = False)->list:
    """
    Cleans a list of text paragraphs using the clean_text function.

    Args:
    corpora (list): A list of text paragraphs to be cleaned.
    count_average_num_word (bool): Count the average number of words for the whole corpora.
    
    Returns:
    list: A list of cleaned text paragraphs.

    Example:
    corpora = ["This is the first paragraph.", "Here is the second one!"]
    result = clean_corpora(corpora)
    # result might be ["this is the first paragraph", "here is the second one"]
    """
    cleaned_corpus = []
    paragraph_sizes = []
    for paragraph in corpus:
        cleaned_paragraph = clean_text(paragraph)
        cleaned_corpus.append(cleaned_paragraph)
        if count_average_num_word:
            paragraph_sizes.append(len(cleaned_paragraph))
    if count_average_num_word:
        print("Average word count:", sum(paragraph_sizes) / len(corpus))
    return cleaned_corpus

def clean_text(text: str) -> list[str]:
    """
    Cleans the input text by tokenizing, filtering, removing stop words, and lemmatizing.

    Args:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned and lemmatized text as a single string.

    Example:
    text = "This is a sample text with <br> HTML tags and stopwords."
    result = clean_text(text)  # result will be ["sample", "text", "html", "tag", "stopword"]
    """
    stop_words = stopwords.words('english')
    stop_words.append("br")
    collected_tokens = []
    for token in word_tokenize(text):
        # lowercasing and removing non word characters
        clean_token = _chr_filter(token)
        # remove stop words
        if clean_token and clean_token not in stop_words:
            collected_tokens.append(clean_token)  

    # lemmatize
    return _lemmatize_with_pos(collected_tokens)

def build_vocab(cleaned_corpus: list[list[str]], vocab_size: int = 500, file_name: str = 'model_dictionary',
                is_save_enabled: bool = False):
    """
    Builds a vocabulary from a list of text paragraphs, with options to save the vocabulary to a file.

    Args:
    corpora (list): A list of text paragraphs to build the vocabulary from.
    vocab_size (int, optional): The desired size of the vocabulary. Default is 500.
    file_name (str, optional): The name of the file to save the vocabulary. Default is 'model_dictionary'.
    is_save_enabled (bool, optional): If True, saves the vocabulary to a file. Default is False.

    Returns:
    list: A list of vocabulary words, including special tokens <unk> and <pad>.

    Example:
    corpus = ["This is the first paragraph.", "Here is the second one!"]
    vocab = build_vocab(corpus, vocab_size=100, file_name='vocab', is_save_enabled=True)
    # vocab is then ['<unk>', '<pad>', 'the', 'is', 'first', 'paragraph', 'here', 'second', 'one']
    """
    # Count word frequency
    word_counts = Counter()
    for paragraph in cleaned_corpus:
        for word in paragraph:
            word_counts[word] += 1
    
    # -2 for making space for unknown words <unk> and padding <pad>
    vocab = []
    vocab.append("<unk>") # needs to be on index position 0 for pytorch training
    vocab.append("<pad>") # needs to be on index position 1 for pytorch training
    # arrange words according to the frequency in decreasing order, only up to vocab_size
    vocab.extend([word for word, count in word_counts.most_common(vocab_size - 2)])

    if vocab_size != len(vocab):
        sys.exit("Can't produce dictionary with given size. Dictionary size: " + str(len(vocab)))

    if is_save_enabled:
        file_utils.save_vocabulary(vocabulary=vocab, vocab_name=file_name)
    return vocab

def build_word_to_idx_vocab(vocab: list)->dict:
    """
    Builds a dictionary mapping each word in the vocabulary to its corresponding index.

    Args:
    vocab (list): A list of words to be included in the vocabulary.

    Returns:
    dict: A dictionary where keys are words from the vocabulary and values are their respective indices.
    """
    return {x: i for i, x in enumerate(vocab)}

def replace_token_to_word_idx(text_list: list, word_dict: iter):
    """
    Converts a list of tokens into their corresponding word indices based on a given word dictionary.

    Args:
    text_list (list): A list of tokens (strings) to be converted.
    word_dict (iter): An iterable containing the vocabulary, where each word corresponds to its index.

    Returns:
    list: A list of indices corresponding to the tokens in the input text_list.

    Example:
    text_list = ['hello', 'world']
    word_dict = ['hello', 'world', 'foo', 'bar']
    result = replace_token_to_word_idx(text_list, word_dict)
    # result will be [0, 1]
    """
    # Convert list to dictionary
    word_to_vec_dict = {value: index for index, value in enumerate(word_dict)}
    paragraph_vector = []
    for token in text_list:
        paragraph_vector.append(word_to_vec_dict[token])
    return paragraph_vector

def get_tf_idf_from_raw_corpora(corpora: list, vocab: Iterable):
    """
    Generates TF-IDF vectors from raw text corpora using a specified vocabulary.

    Args:
    corpora (list): A list of raw text paragraphs to be processed.
    vocab (Iterable): An iterable containing the vocabulary to be used for TF-IDF vectorization.

    Returns:
    Any: The TF-IDF vectors generated from the input corpora.

    Example:
    corpora = ["This is the first paragraph.", "Here is the second one!"]
    vocab = ["this", "is", "the", "first", "paragraph", "here", "second", "one"]
    tf_idf_vectors = get_tf_idf_from_raw_corpora(corpora, vocab)
    # tf_idf_vectors will contain the TF-IDF representation of the input corpora
    """
    print(f"Generating TF-IDF vectors in size: {len(vocab)} from corpora.")
    collected_paragraph = []
    for paragraph in corpora:
        cleaned_text = clean_text(paragraph)
        collected_paragraph.append(cleaned_text)
    return vectorizers.build_tf_idf_from_corpora(corpora=collected_paragraph,
                                                        vocab=vocab)

def get_tf_from_raw_corpora(corpora: list, vocab: Iterable, is_one_hot: bool = False):
    """
    Generates Term Frequency (TF) vectors from raw text corpora using a specified vocabulary.

    Args:
    corpora (list): A list of raw text paragraphs to be processed.
    vocab (Iterable): An iterable containing the vocabulary to be used for TF vectorization.
    is_one_hot (bool, optional): If True, generates one-hot encoded vectors. Default is False.

    Returns:
    Any: The TF vectors generated from the input corpora.

    Example:
    corpora = ["This is the first paragraph.", "Here is the second one!"]
    vocab = ["this", "is", "the", "first", "paragraph", "here", "second", "one"]
    tf_vectors = get_tf_from_raw_corpora(corpora, vocab, is_one_hot=False)
    # tf_vectors will contain the TF representation of the input corpora
    """
    print(f"Generating TF vectors in size: {len(vocab)} from corpora.")
    collected_paragraph = []
    for paragraph in corpora:
        cleaned_text = clean_text(paragraph)
        collected_paragraph.append(cleaned_text)
    return vectorizers.build_tf_from_corpora(corpora=collected_paragraph,
                                                        vocab=vocab,
                                                        is_one_hot=is_one_hot)

def build_glo_ve_model(dataset_name:str, vector_size:int = 100, window_size: int=5, workers: int=3, 
                       vocab_size: int = 20000 ):
    """
    Trains a GloVe model using a dataset from a CSV file.

    Args:
    dataset_name (str): The path to the input CSV file containing the dataset.
    vector_size (int, optional): The dimensionality of the word vectors. Default is 100.
    window_size (int, optional): The maximum distance between the current and predicted word within a sentence. Default is 5.
    workers (int, optional): The number of worker threads to use for training. Default is 3.
    vocab_size (int, optional): The maximum size of the vocabulary. Default is 20000.

    Returns:
    Word2Vec: The trained GloVe model.

    Example:
    model = build_glo_ve_model('reviews.csv', vector_size=200, window_size=10, workers=4, vocab_size=30000)
    # model will be a trained GloVe model with the specified parameters

    Notes:
    This function is inspired by the Gensim tutorial:
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
    """
    df = pd.read_csv(dataset_name)
    cleaned_train_corpora = clean_corpus(corpora=df['review'].tolist())
    corpora = []
    for line in cleaned_train_corpora:
        # one paragraph in a line, separated by a simple space for the processor
        corpora.append(utils.simple_preprocess(line))
    # -1 to add <unk> token with zero vector
    return Word2Vec(sentences=corpora, vector_size=vector_size, window=window_size, sg=1, 
                    min_count=1, max_final_vocab=vocab_size - 1)

def build_token_list(cleaned_corpora: list):
    """
    Converts a list of cleaned text paragraphs into a list of token lists.

    Args:
    cleaned_corpora (list): A list of cleaned text paragraphs.

    Returns:
    list: A list where each element is a list of tokens from the corresponding paragraph.

    Example:
    cleaned_corpora = ["this is a cleaned paragraph", "another cleaned paragraph"]
    result = build_token_list(cleaned_corpora)
    # result will be [['this', 'is', 'a', 'cleaned', 'paragraph'], ['another', 'cleaned', 'paragraph']]
    """
    corpora_list = []
    for paragraph in cleaned_corpora:
        paragraph_list = []
        for token in word_tokenize(paragraph):
            paragraph_list.append(token)
        corpora_list.append(paragraph_list)
    return corpora_list

