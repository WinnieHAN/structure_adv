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
# from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvCRF, BiVarRecurrentConvCRF
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter

from seq2seq_rl.seq2seq import Seq2seq_Model
# import pickle
from nltk.tag.senna import SennaTagger
from nltk.tag import StanfordPOSTagger


def parse_diff(out, dec_out, length_out):
    stc_dda = sum([0 if out[i] == dec_out[i] else 1 for i in range(1, length_out)])
    return float(stc_dda) / float(length_out - 1)


def unk_rate(sent):
    return float(sent.count(0)) / float(len(sent))


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN-CRF')
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', required=True)
    parser.add_argument('--cuda', action='store_true', help='using GPU')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN')
    parser.add_argument('--tag_space', type=int, default=0, help='Dimension of tag space')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')  # 30
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    parser.add_argument('--dropout', choices=['std', 'variational'], help='type of dropout', required=True)
    parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    parser.add_argument('--bigram', action='store_true', help='bi-gram parameter for CRF')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    parser.add_argument('--seq2seq_load_path', default='tagging_models/tagging/seq2seq/seq2seq_save_model', type=str, help='seq2seq_load_path')
    parser.add_argument('--network_load_path', default='tagging_models/tagging/seq2seq/network_save_model', type=str, help='network_load_path')

    parser.add_argument('--rl_finetune_seq2seq_save_path', default='tagging_models/tagging/rl_finetune/seq2seq_save_model', type=str, help='rl_finetune_seq2seq_save_path')
    parser.add_argument('--rl_finetune_network_save_path', default='tagging_models/tagging/rl_finetune/network_save_model', type=str, help='rl_finetune_network_save_path')

    parser.add_argument('--rl_finetune_seq2seq_load_path', default='tagging_models/tagging/rl_finetune/seq2seq_save_model', type=str, help='rl_finetune_seq2seq_load_path')
    parser.add_argument('--rl_finetune_network_load_path', default='tagging_models/tagging/rl_finetune/network_save_model', type=str, help='rl_finetune_network_load_path')

    parser.add_argument('--treebank', type=str, default='ctb', help='tree bank', choices=['ctb', 'ptb'])  # ctb

    parser.add_argument('--direct_eval', action='store_true', help='direct eval without generation process')
    parser.add_argument('--port', type=int, default=10048, help='localhost port for berscore server')
    parser.add_argument('--z1_weight', type=float, default=1.0, help='reward weight of z1')
    parser.add_argument('--z2_weight', type=float, default=1.0, help='reward weight of z2')
    parser.add_argument('--z3_weight', type=float, default=0.01, help='reward weight of z3')
    parser.add_argument('--mp_weight', type=float, default=100, help='reward weight of meaning preservation')
    parser.add_argument('--ppl_weight', type=float, default=0.001, help='reward weight of ppl')
    parser.add_argument('--unk_weight', type=float, default=1000, help='reward weight of unk rate')
    parser.add_argument('--prefix', action='append')
    parser.add_argument('--parserb', type=str, required=True)
    parser.add_argument('--parserc', type=str, required=True)


    args = parser.parse_args()

    logger = get_logger("POSCRFTagger")

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    bigram = args.bigram

    embedding = args.embedding
    embedding_path = args.embedding_dict

    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)

    parser_select = ['tagger0', 'tagger1']

    parser_b = args.parserb
    parser_c = args.parserc

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("data/alphabets/pos_crf/", train_path,data_paths=[dev_path, test_path],
                                                 max_vocabulary_size=50000, embedd_dict=embedd_dict)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    logger.info("Reading Data")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    data_train = conllx_data.read_data_to_tensor(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, device=device)
    # data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    # num_data = sum([len(bucket) for bucket in data_train])
    num_data = sum(data_train[1])
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


    shared_word_embedd = network.return_word_embedd()
    shared_word_embedd.weight.requires_grad = False
    num_words = word_alphabet.size()
    seq2seq = Seq2seq_Model(EMB=embedd_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=shared_word_embedd, device=device).to(device)  # debug hanwj
    seq2seq.emb.weight.requires_grad = False
    print(seq2seq)


    sudo_golden_tagger = SennaTagger(parser_b)
    sudo_golden_tagger_1 = StanfordPOSTagger(model_filename=parser_c + '/models/english-bidirectional-distsim.tagger',
                                             path_to_jar=parser_c + '/stanford-postagger.jar')

    def word_to_chars_tensor(shape, sel, lengths_sel, word_alphabet, char_alphabet):
        batch_s, stc_length, char_length = shape
        char_length = 40
        chars = np.ones(shape=[batch_s, stc_length, char_length], dtype=int)
        for si in range(batch_s):
            for wi in range(lengths_sel[si]):
                word_str = word_alphabet.get_instance(sel[si, wi])
                temp = [char_alphabet.get_index(charstr) for charstr in word_str]
                chars[si, wi] = temp[0:char_length] + [1 for _ in range(
                    char_length - len(temp))]  # temp + np.ones(char_length-len(temp), dtype=int)
        return torch.from_numpy(chars).to(device) #torch.tensor(chars, dtype=int)

    END_token = word_alphabet.instance2index['_PAD']


    seq2seq.eval()
    network.eval()
    for prefix in args.prefix:
        src_filename_test = 'tagging_dumped/' + '_' + prefix + 'src_test'

        # load pretrained model
        seq2seq.load_state_dict(torch.load(args.rl_finetune_seq2seq_load_path + prefix + '.pt'))
        # seq2seq.load_state_dict(torch.load('/media/zhanglw/2EF0EF5BF0EF27B3/code/rl_finetune/seq2seq_save_model_' + prefix + '.pt'))
        seq2seq.to(device)
        network.load_state_dict(torch.load(args.rl_finetune_network_load_path + prefix + '.pt'))
        # network.load_state_dict(torch.load('/media/zhanglw/2EF0EF5BF0EF27B3/code/rl_finetune/network_save_model_'+ prefix + '.pt'))
        network.to(device)

        src_writer_test = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        src_writer_test.start(src_filename_test)

        kk = 0
        batch_size_for_eval = 32
        ab_diff = 0
        ac_diff = 0
        bc_same = 0
        cnt = 0
        generation_res = []
        for batch in conllx_data.iterate_batch_tensor(data_train, batch_size_for_eval):  # batch_size
            print('----' + str(kk) + '----')
            kk += 1
            word, char, labels, _, _, masks, lengths = batch
            with torch.no_grad():
                if not args.direct_eval:
                    inp = word
                    sel, pb = seq2seq(inp.long().to(device), LEN=inp.size()[1])
                    end_position = torch.eq(sel, END_token).nonzero()
                    masks_sel = torch.ones_like(sel, dtype=torch.float)
                    lengths_sel = torch.ones_like(lengths).fill_(
                        sel.shape[1])
                    if not len(end_position) == 0:
                        ij_back = -1
                        for ij in end_position:
                            if not (ij[0] == ij_back):
                                lengths_sel[ij[0]] = ij[1]
                                masks_sel[ij[0], ij[1]:] = 0
                                ij_back = ij[0]
                    char1 = word_to_chars_tensor(char.shape, sel, lengths_sel, word_alphabet, char_alphabet)
                else:
                    sel = word
                    pb = torch.ones_like(sel, dtype=torch.float).fill_(0)
                    lengths_sel = lengths
                    masks_sel = masks
                    char1 = char
                tags_pred, corr = network.decode(sel, char1, target=labels, mask=masks_sel, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                sel_num = sel.cpu().numpy()
                sentence_max_length = len(sel_num[0])
                str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in
                            sel_num[one_stc_id][:lengths_sel[one_stc_id]]] for
                           one_stc_id in range(len(sel_num))]
                if 'tagger0' in parser_select:
                    temp = [[pos_alphabet.get_index(j[1]) for j in sudo_golden_tagger.tag(i)] for i in str_sel]
                    sudo_tags_pred = [i + [1 for _ in range(sentence_max_length - len(i))] for i in temp]
                if 'tagger1' in parser_select:
                    temp = [[pos_alphabet.get_index(j[1]) for j in sudo_golden_tagger_1.tag(i)] for i in str_sel]
                    sudo_tags_pred_1 = [i + [1 for _ in range(sentence_max_length - len(i))] for i in temp]

            lengths_sel = lengths_sel.detach().cpu().numpy()

            for i in range(batch[0].size()[0]):
                cnt += 1
                parse_ab = False
                if parse_diff(tags_pred[i], sudo_tags_pred[i], lengths_sel[i]) != 0:
                    parse_ab = True
                    ab_diff += 1
                parse_ac = False
                if parse_diff(tags_pred[i], sudo_tags_pred_1[i], lengths_sel[i]) != 0:
                    parse_ac = True
                    ac_diff += 1
                parse_bc = False
                if parse_diff(sudo_tags_pred[i], sudo_tags_pred_1[i], lengths_sel[i]) == 0:
                    parse_bc = True
                    bc_same += 1
                if parse_ab and parse_ac and parse_bc:
                    generation_res.append((word[i].cpu().numpy().tolist(),
                                           sel[i].cpu().numpy().tolist(),
                                           tags_pred[i].tolist(),
                                           sudo_tags_pred[i].tolist(),
                                           lengths_sel[i].item()))
                else:
                    pass

        for word, sel, pred, golden, length in generation_res:
            head = [0] * length
            src_writer_test.write(word, golden, head, None, length)

        src_writer_test.close()

        print('==' * 10 + prefix + '==' * 10)
        print('Total cnt: ' + str(cnt))
        print("-" * 10)
        print('generate data: ' + str(len(generation_res)))
        print("-" * 10)
        print('ab difference data: ' + str(ab_diff))
        print("-" * 10)
        print('ac difference data: ' + str(ac_diff))
        print("-" * 10)
        print('bc same data: ' + str(bc_same))


if __name__ == '__main__':
    main()
