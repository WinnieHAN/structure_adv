from __future__ import print_function

import global_variables
from neuronlp2 import utils
from neuronlp2.io import conllx_data, CoNLLXWriter
from seq2seq_rl.rl import LossBiafRL
import numpy as np

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os

import torch
reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.append(".")
sys.path.append("..")

import uuid

uid = uuid.uuid4().hex[:6]


# 3 sub-models should be pretrained in our approach
#   seq2seq pretrain, denoising autoencoder  | or using token-wise adv to generate adv examples.
#   structure prediction model
#   oracle parser
# then we train the seq2seq model using rl


def main():
    model_path = "./models/parsing/biaffine/"
    alphabet_path = os.path.join(model_path, 'alphabets/')
    word_embedding = "sskip"
    word_path = "./data/sskip/sskip.eng.100.gz"
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, None,
                                                                                             data_paths=[None, None],
                                                                                             max_vocabulary_size=100000,
                                                                                             embedd_dict=word_dict)
    train_path = '/media/zhanglw/2EF0EF5BF0EF27B3/code/parsing/report_result/b16_unk500_parserbc_lr2e4_pred_test3_parseA.txt'
    data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    device = torch.device('cuda')
    num_words = word_alphabet.size()

    global_variables.PREFIX = "/home/zhanglw/code/rebase/structure_adv/test_"

    rl = LossBiafRL(device=device, word_alphabet=word_alphabet, vocab_size=num_words, port=10048).to(device)

    pred_filename_test = './dumped/human_score'

    pred_writer_test = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    pred_writer_test.start(pred_filename_test)

    with torch.no_grad():
        cnt = 0
        for batch in conllx_data.iterate_batch(data_train, batch_size=20):

            word, char, pos, heads, types, masks = batch
            lengths = np.sum(masks, axis=1)
            meaning_preservation, logppl = rl.get_bertscore_ppl(word, lengths, word, lengths)
            if logppl.shape == ():
                logppl = np.array([logppl, logppl])
            pred_writer_test.write_stc(word, lengths.astype(np.int32), symbolic_root=True, ppl=logppl)
            print(cnt)
            cnt += 1


if __name__ == '__main__':

    main()
