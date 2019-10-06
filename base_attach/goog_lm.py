"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import os
import tensorflow as tf
import sys
import numpy as np

from google.protobuf import text_format

import lm_utils
import lm_data_utils


class LM(object):
    def __init__(self):
        self.PBTXT_PATH = 'goog_lm/graph-2016-09-10.pbtxt'
        self.CKPT_PATH = 'goog_lm/ckpt-*'
        self.VOCAB_PATH = 'goog_lm/vocab-2016-09-10.txt'

        self.BATCH_SIZE = 1
        self.NUM_TIMESTEPS = 1
        self.MAX_WORD_LEN = 50

        self.vocab = lm_data_utils.CharsVocabulary(self.VOCAB_PATH, self.MAX_WORD_LEN)
        print('LM vocab loading done')
        with tf.device("/gpu:1"):
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.t = lm_utils.LoadModel(self.sess, self.graph, self.PBTXT_PATH, self.CKPT_PATH)



    def get_words_probs(self, prefix_words, list_words, suffix=None):
        targets = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        weights = np.ones([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.float32)

        if prefix_words.find('<S>') != 0:
            prefix_words = '<S> ' + prefix_words
        prefix = [self.vocab.word_to_id(w) for w in prefix_words.split()]
        prefix_char_ids = [self.vocab.word_to_char_ids(w) for w in prefix_words.split()]

        inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS, self.vocab.max_word_length], np.int32)

        samples = prefix[:]
        char_ids_samples = prefix_char_ids[:]
        inputs = [ [samples[-1]]]
        char_ids_inputs[0, 0, :] = char_ids_samples[-1]
        softmax = self.sess.run(self.t['softmax_out'],
        feed_dict={
            self.t['char_inputs_in']: char_ids_inputs,
            self.t['inputs_in']: inputs,
            self.t['targets_in']: targets,
            self.t['target_weights_in']: weights
        })
        # print(list_words)
        words_ids = [self.vocab.word_to_id(w) for w in list_words]
        word_probs =[softmax[0][w_id] for w_id in words_ids]
        word_probs = np.array(word_probs)

        if suffix == None:
            suffix_probs = np.ones(word_probs.shape)
        else:
            suffix_id = self.vocab.word_to_id(suffix)
            suffix_probs = []
            for idx, w_id in enumerate(words_ids):
                # print('..', list_words[idx])
                inputs = [[w_id]]
                w_char_ids = self.vocab.word_to_char_ids(list_words[idx])
                char_ids_inputs[0, 0, :] = w_char_ids
                softmax = self.sess.run(self.t['softmax_out'],
                                         feed_dict={
                                             self.t['char_inputs_in']: char_ids_inputs,
                                             self.t['inputs_in']: inputs,
                                             self.t['targets_in']: targets,
                                             self.t['target_weights_in']: weights
                                         })
                suffix_probs.append(softmax[0][suffix_id])
            suffix_probs = np.array(suffix_probs)
        # print(word_probs, suffix_probs)
        return suffix_probs * word_probs


if __name__ == '__main__':
   my_lm = LM()
   list_words = 'play will playing played afternoon'.split()
   prefix = 'i'
   suffix = 'yesterday'
   probs = (my_lm.get_words_probs(prefix, list_words, suffix))
   for i, w in enumerate(list_words):
       print(w, ' - ', probs[i])
