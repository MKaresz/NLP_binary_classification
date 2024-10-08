from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Iterable
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import numpy as np


def build_tf_from_corpora(corpora: list,
                          vocab: Iterable,
                          is_one_hot: bool = False
                          ):
    """
    Generates Term Frequency (TF) vectors from a list of text corpora using a specified vocabulary.

    Args:
    corpora (list): A list of text documents to be vectorized.
    vocab (Iterable): An iterable containing the vocabulary to be used for TF vectorization.
    is_one_hot (bool, optional): If True, generates one-hot encoded vectors. Default is False.

    Returns:
    np.ndarray: An array of TF vectors generated from the input corpora.

    Example:
    corpora = ["This is the first document.", "This document is the second document."]
    vocab = ["this", "is", "the", "first", "document", "second"]
    tf_vectors = build_tf_from_corpora(corpora, vocab, is_one_hot=False)
    # tf_vectors will contain the TF representation of the input corpora
    """
    vectorizer = CountVectorizer(vocabulary=vocab, binary=is_one_hot)
    return vectorizer.fit_transform(corpora).toarray()

def build_tf_idf_from_corpora(corpora: list, vocab: Iterable):
    """
    Generates TF-IDF vectors from a list of text corpora using a specified vocabulary.

    Args:
    corpora (list): A list of text documents to be vectorized.
    vocab (Iterable): An iterable containing the vocabulary to be used for TF-IDF vectorization.

    Returns:
    np.ndarray: An array of TF-IDF vectors generated from the input corpora.

    Example:
    corpora = ["This is the first document.", "This document is the second document."]
    vocab = ["this", "is", "the", "first", "document", "second"]
    tf_idf_vectors = build_tf_idf_from_corpora(corpora, vocab)
    # tf_idf_vectors will contain the TF-IDF representation of the input corpora
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab, binary=False)
    return vectorizer.fit_transform(corpora).toarray()

def build_word_to_idx_vector(corpus: list, vocab_idx: dict, vec_size: int):
    """
    Converts a list of texts into a list of fixed-size word index vectors.

    Args:
    corpora (list): A list of text documents to be converted.
    vocab_idx (dict): A dictionary mapping words to their corresponding indices.
    vec_size (int): The fixed size of the output vectors.

    Returns:
    list: A list of word index vectors, each of fixed size `vec_size`.

    Example:
    corpora = ["This is a sentence.", "Another sentence here."]
    vocab_idx = {"this": 2, "is": 3, "a": 4, "sentence": 5, "another": 6, "here": 7}
    vec_size = 10
    result = build_word_to_idx_vector(corpora, vocab_idx, vec_size)
    # result might be [[2, 3, 4, 5, 1, 1, 1, 1, 1, 1], [6, 5, 7, 1, 1, 1, 1, 1, 1, 1]]
    """
    tokenized_corpus = []
    for paragraph in corpus:
        paragraph_vec = []
        for token in paragraph:
            # unknown word <unk> is on 0 index
            paragraph_vec.append(vocab_idx.get(token, 0))
        # add padding and create the fixed size vector
        paragraph_vec.extend([1] * (vec_size - len(paragraph_vec)))
        tokenized_corpus.append(paragraph_vec[:vec_size])
    return tokenized_corpus

def build_idx_to_word_vector(corpora_vec: list, vocab_idx: dict):
    """
    Converts a list of word index vectors back into a list of word lists using a vocabulary index.

    Args:
    corpora_vec (list): A list of word index vectors.
    vocab_idx (dict): A dictionary mapping words to their corresponding indices.

    Returns:
    list: A list of word lists, where each word list corresponds to the original word index vector.

    Example:
    corpora_vec = [[2, 3, 4, 5], [6, 5, 7]]
    vocab_idx = {"this": 2, "is": 3, "a": 4, "sentence": 5, "another": 6, "here": 7}
    result = build_idx_to_word_vector(corpora_vec, vocab_idx)
    # result might be [['this', 'is', 'a', 'sentence'], ['another', 'sentence', 'here']]
    """
    tokenized_corpora = []
    for idx_list in corpora_vec:
        paragraph_words = []
        for idx in idx_list:
            for key, value in vocab_idx.items():
                if value is idx: paragraph_words.append(key)
        tokenized_corpora.append(paragraph_words)
    return tokenized_corpora

def build_glo_ve_vector(corpora_tokens: list, wv: KeyedVectors, paragraph_size:int = 300):
    """
    Converts a list of tokenized paragraphs into GloVe vectors with a fixed paragraph size.

    Args:
    corpora_tokens (list): A list of tokenized paragraphs.
    wv (KeyedVectors): The GloVe word vectors.
    paragraph_size (int, optional): The fixed size of the output paragraph vectors. Default is 300.

    Returns:
    list: A list of paragraphs represented as GloVe vectors, each of fixed size `paragraph_size`.

    Example:
    corpora_tokens = [["this", "is", "a", "sentence"], ["another", "sentence", "here"]]
    wv = load_glo_ve_vector('my_glove_vector')
    result will be a list of paragraphs represented as GloVe vectors of size 100 dim
    """
    corpora_vec = []
    for paragraph in corpora_tokens:
        paragraph_vec = []
        for word in paragraph:
            if word in wv:
                paragraph_vec.append(wv.get_vector(word))
            else:
                paragraph_vec.append(np.zeros(shape=(wv.vector_size,), dtype=np.float32))
        paragraph_size_difference = paragraph_size-len(paragraph_vec)

        if paragraph_size_difference < 0:
            paragraph_vec = paragraph_vec[:paragraph_size_difference]
        elif paragraph_size_difference > 0:
            paragraph_vec.extend([np.zeros(shape=(wv.vector_size,), dtype=np.float32)] * paragraph_size_difference)
        if len(paragraph_vec) != paragraph_size:
            import sys
            sys.exit("Paragraph size error in vectorizers.build_glo_ve_vector()!")
        corpora_vec.append(paragraph_vec)
    return corpora_vec

