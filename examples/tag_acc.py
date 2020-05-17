# Tagging accurate for model


from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for POS tagging.
"""

import sys

sys.path.append(".")
sys.path.append("..")

# import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvCRF, BiVarRecurrentConvCRF
from neuronlp2 import utils
from nltk.tag.senna import SennaTagger
from nltk.tag import StanfordPOSTagger

def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN-CRF')
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', default='LSTM')
    parser.add_argument('--cuda', action='store_true', help='using GPU')
    parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    parser.add_argument('--tag_space', type=int, default=256, help='Dimension of tag space')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')  # 30
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    parser.add_argument('--dropout', choices=['std', 'variational'], help='type of dropout', default='std')
    parser.add_argument('--p_rnn', nargs=2, type=float, default=[0.33, 0.5], help='dropout rate for RNN')
    parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    parser.add_argument('--bigram', action='store_true', help='bi-gram parameter for CRF')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', default='sskip')
    parser.add_argument('--embedding_dict', help='path for embedding dict', default='data/sskip/sskip.eng.100.gz')
    parser.add_argument('--train', default='data/ptb/train.conllu')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev', default='data/ptb/dev.conllu')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test', default='data/ptb/test.conllu')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    parser.add_argument('--parser', choices=['bicrf', 'senna', 'stanford'], default='stanford')

    args = parser.parse_args()

    logger = get_logger("POSCRFTagger")

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    hidden_size = args.hidden_size
    num_filters = args.num_filters
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    bigram = args.bigram

    embedding = args.embedding
    embedding_path = args.embedding_dict

    parser = args.parser
    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("data/alphabets/pos_crf/", train_path,data_paths=[dev_path, test_path],
                                                 max_vocabulary_size=50000, embedd_dict=embedd_dict)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    logger.info("Reading Data")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    # data_train = conllx_data.read_data_to_tensor(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, device=device)

    # num_data = sum(data_train[1])
    num_labels = pos_alphabet.size()

    data_dev = conllx_data.read_data_to_tensor(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, device=device)
    data_test = conllx_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, device=device)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in embedd_dict:
                embedding = embedd_dict[word]
            elif word.lower() in embedd_dict:
                embedding = embedd_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")

    if parser == 'bicrf':
        char_dim = args.char_dim
        window = 3
        num_layers = args.num_layers
        tag_space = args.tag_space
        initializer = nn.init.xavier_uniform_
        if args.dropout == 'std':
            network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters, window, mode, hidden_size, num_layers, num_labels,
                                         tag_space=tag_space, embedd_word=word_table, bigram=bigram, p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer)
        else:
            network = BiVarRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters, window, mode, hidden_size, num_layers, num_labels,
                                            tag_space=tag_space, embedd_word=word_table, bigram=bigram, p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer)


    # shared_word_embedd = network.return_word_embedd()
    # shared_word_embedd.weight.requires_grad = False
    # num_words = word_alphabet.size()



        network.load_state_dict(torch.load('tagging_models/tagging/seq2seq/network_save_model' + str(2) + '.pt'))
        network.to(device)

        # evaluate performance on dev data
        with torch.no_grad():
            network.eval()
            dev_corr = 0.0
            dev_total = 0
            for batch in conllx_data.iterate_batch_tensor(data_dev, 32):
                word, char, labels, _, _, masks, lengths = batch
                preds, corr = network.decode(word, char, target=labels, mask=masks,
                                             leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                num_tokens = masks.sum()
                dev_corr += corr
                dev_total += num_tokens

            print('dev corr: %d, total: %d, acc: %.2f%%' % (dev_corr, dev_total, dev_corr * 100 / dev_total))


            # evaluate on test data when better performance detected
            test_corr = 0.0
            test_total = 0
            for batch in conllx_data.iterate_batch_tensor(data_test, 32):
                word, char, labels, _, _, masks, lengths = batch
                preds, corr = network.decode(word, char, target=labels, mask=masks,
                                             leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                num_tokens = masks.sum()
                test_corr += corr
                test_total += num_tokens

            test_correct = test_corr
            print("best dev  corr: %d, total: %d, acc: %.2f%%" % (dev_corr, dev_total, dev_corr * 100 / dev_total))
            print("best test corr: %d, total: %d, acc: %.2f%%" % (test_correct, test_total, test_correct * 100 / test_total))
    elif parser == 'senna':
        network = SennaTagger('tagging_models/senna')
        dev_corr = 0.0
        dev_total = 0

        for batch in conllx_data.iterate_batch_tensor(data_dev, 32):
            print(dev_corr)
            word, char, labels, _, _, masks, lengths = batch
            labels = labels.cpu().numpy()
            sel_num = word.cpu().numpy()
            lengths = lengths.cpu().numpy()
            str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in
                        sel_num[one_stc_id][:lengths[one_stc_id]]] for
                       one_stc_id in range(len(sel_num))]
            temp = [[pos_alphabet.get_index(j[1]) for j in network.tag(i)] for i in str_sel]
            dev_total += np.sum(lengths)

            for i in range(labels.shape[0]):
                for j in range(len(temp[i])):
                    if temp[i][j] == labels[i][j]:
                        dev_corr += 1
        print("dev  corr: %d, total: %d, acc: %.2f%%" % (dev_corr, dev_total, dev_corr * 100 / dev_total))

        test_corr = 0.0
        test_total = 0
        for batch in conllx_data.iterate_batch_tensor(data_test, 32):
            print(test_corr)
            word, char, labels, _, _, masks, lengths = batch
            labels = labels.cpu().numpy()
            sel_num = word.cpu().numpy()
            lengths = lengths.cpu().numpy()
            str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in
                        sel_num[one_stc_id][:lengths[one_stc_id]]] for
                       one_stc_id in range(len(sel_num))]
            temp = [[pos_alphabet.get_index(j[1]) for j in network.tag(i)] for i in str_sel]
            test_total += np.sum(lengths)

            for i in range(labels.shape[0]):
                for j in range(len(temp[i])):
                    if temp[i][j] == labels[i][j]:
                        test_corr += 1
        print("test  corr: %d, total: %d, acc: %.2f%%" % (test_corr, test_total, test_corr * 100 / test_total))

    elif parser == 'stanford':
        network = StanfordPOSTagger(model_filename='tagging_models/stanford-postagger/models/english-bidirectional-distsim.tagger',
                                                 path_to_jar='tagging_models/stanford-postagger/stanford-postagger.jar')
        dev_corr = 0.0
        dev_total = 0
        for batch in conllx_data.iterate_batch_tensor(data_dev, 32):
            word, char, labels, _, _, masks, lengths = batch
            labels = labels.cpu().numpy()
            sel_num = word.cpu().numpy()
            lengths = lengths.cpu().numpy()
            str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in
                        sel_num[one_stc_id][:lengths[one_stc_id]]] for
                       one_stc_id in range(len(sel_num))]
            temp = [[pos_alphabet.get_index(j[1]) for j in network.tag(i)] for i in str_sel]
            dev_total += np.sum(lengths)

            for i in range(labels.shape[0]):
                for j in range(len(temp[i])):
                    if temp[i][j] == labels[i][j]:
                        dev_corr += 1
        print("dev  corr: %d, total: %d, acc: %.2f%%" % (dev_corr, dev_total, dev_corr * 100 / dev_total))

        test_corr = 0.0
        test_total = 0
        for batch in conllx_data.iterate_batch_tensor(data_test, 32):
            word, char, labels, _, _, masks, lengths = batch
            labels = labels.cpu().numpy()
            sel_num = word.cpu().numpy()
            lengths = lengths.cpu().numpy()
            str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in
                        sel_num[one_stc_id][:lengths[one_stc_id]]] for
                       one_stc_id in range(len(sel_num))]
            temp = [[pos_alphabet.get_index(j[1]) for j in network.tag(i)] for i in str_sel]
            test_total += np.sum(lengths)

            for i in range(labels.shape[0]):
                for j in range(len(temp[i])):
                    if temp[i][j] == labels[i][j]:
                        test_corr += 1
        print("test  corr: %d, total: %d, acc: %.2f%%" % (test_corr, test_total, test_corr * 100 / test_total))

if __name__ == '__main__':
    main()
