from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib.request
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf

window_size = 3
vector_dim = 300
epochs = 1000

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

def relu(x):
    """
    Compute the relu of x
    """
    s = np.maximum(0,x)
    
    return s


def initialize_parameters(vocab_size, n_h):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2":
                    W1 -- weight matrix of shape (n_h, vocab_size)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (vocab_size, n_h)
                    b2 -- bias vector of shape (vocab_size, 1)
    """
    
    np.random.seed(3)
    parameters = {}

    parameters['W1'] = np.random.randn(n_h, vocab_size) / np.sqrt(vocab_size)
    parameters['b1'] = np.zeros((n_h, 1))
    parameters['W2'] = np.random.randn(vocab_size, n_h) / np.sqrt(n_h)
    parameters['b2'] = np.zeros((vocab_size, 1))

    return parameters

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
