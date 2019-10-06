"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""

import numpy as np
import pickle


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        # print(word)
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print ("Done.", len(model), " words loaded!")
    return model


def save_glove_to_pickle(glove_model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(glove_model, f)


def load_glove_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def create_embeddings_matrix(glove_model, dictionary, full_dictionary, d=300):
    MAX_VOCAB_SIZE = len(dictionary)
    # Matrix size is 300
    embedding_matrix = np.zeros(shape=((d, MAX_VOCAB_SIZE + 1)))
    cnt = 0
    unfound = []

    for w, i in dictionary.items():
        if not w in glove_model:
            cnt += 1
            # if cnt < 10:
            # embedding_matrix[:,i] = glove_model['UNK']
            unfound.append(i)
        else:
            embedding_matrix[:, i] = glove_model[w]
    print('Number of not found words = ', cnt)
    return embedding_matrix, unfound


def pick_most_similar_words(src_word, dist_mat, ret_count=10, threshold=None):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    dist_order = np.argsort(dist_mat[src_word, :])[1:1 + ret_count]
    dist_list = dist_mat[src_word][dist_order]
    if dist_list[-1] == 0:
        return [], []
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        return dist_order[mask], dist_list[mask]
    else:
        return dist_order, dist_list

def pick_most_similar_words_from_vector(src_word, word_vector, dist_mat, ret_count=10, threshold=None):
    dist_order = np.argsort(dist_mat)[1:1 + ret_count]
    dist_list = dist_mat[dist_order]
    if dist_list[-1] == 0:
        return [], []
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        if list(mask[0]):
            return dist_order[mask], dist_list[mask]
        else:
            return [src_word], [0]
    else:
        return dist_order, dist_list