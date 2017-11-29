from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid

import numpy as np
import torch
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.models import StackPtrNet
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

uid = uuid.uuid4().get_hex()[:6]


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU', 'FastLSTM'], help='architecture of rnn',
                             required=True)
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--type_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    args_parser.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_parser.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_parser.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--eta', type=float, default=1.0, help='weight for coverage loss')
    args_parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_parser.add_argument('--left2right', action='store_true', help='apply left to right prior order.')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0.,
                             help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'],
                             help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters',
                             required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args = args_parser.parse_args()

    logger = get_logger("PtrParser")

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    num_layers = args.num_layers
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    momentum = 0.9
    betas = (0.9, 0.9)
    decay_rate = args.decay_rate
    gamma = args.gamma
    eta = args.eta
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    left2right = args.left2right
    beam = args.beam
    punctuation = args.punctuation

    word_embedding = args.word_embedding
    word_path = args.word_path
    char_embedding = args.char_embedding
    char_path = args.char_path

    pos_dim = args.pos_dim
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    char_dict = None
    char_dim = args.char_dim
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)
    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_stacked_data.create_alphabets("data/alphabets/ptr/", train_path,
                                                         data_paths=[dev_path, test_path],
                                                         max_vocabulary_size=50000, embedd_dict=word_dict)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    data_train = conllx_stacked_data.read_stacked_data_to_variable(train_path, word_alphabet, char_alphabet,
                                                                   pos_alphabet, type_alphabet, use_gpu=use_gpu,
                                                                   left2right=left2right)
    num_data = sum(data_train[1])

    data_dev = conllx_stacked_data.read_stacked_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet,
                                                                 type_alphabet, use_gpu=use_gpu, volatile=True,
                                                                 left2right=left2right)
    data_test = conllx_stacked_data.read_stacked_data_to_variable(test_path, word_alphabet, char_alphabet,
                                                                  pos_alphabet, type_alphabet,
                                                                  use_gpu=use_gpu, volatile=True,
                                                                  left2right=left2right)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()

    window = 3
    network = StackPtrNet(word_dim, num_words,
                          char_dim, num_chars,
                          pos_dim, num_pos,
                          num_filters, window,
                          mode, hidden_size, num_layers,
                          num_types, arc_space, type_space,
                          embedd_word=word_table, embedd_char=char_table,
                          p_in=p_in, p_out=p_out, p_rnn=p_rnn, biaffine=True)

    if use_gpu:
        network.cuda()

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    adam_epochs = 50
    adam_rate = 0.001
    if adam_epochs > 0:
        lr = adam_rate
        opt = 'adam'
        optim = Adam(network.parameters(), lr=adam_rate, betas=betas, weight_decay=gamma)
    else:
        opt = 'sgd'
        lr = learning_rate
        optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

    logger.info("Embedding dim: word=%d, char=%d, pos=%d" % (word_dim, char_dim, pos_dim))
    logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, arc_space=%d, type_space=%d" % (
        mode, num_layers, hidden_size, num_filters, arc_space, type_space))
    logger.info("train: cov: %.1f, (#data: %d, batch: %d, dropout(in, out, rnn): (%.2f, %.2f, %s), unk_repl: %.2f)" % (
        eta, num_data, batch_size, p_in, p_out, p_rnn, unk_replace))
    logger.info('prior order: %s, beam: %d' % ('left2right' if left2right else 'inside-out', beam))

    num_batches = num_data / batch_size + 1
    dev_ucorrect = 0.0
    dev_lcorrect = 0.0
    dev_ucomlpete_match = 0.0
    dev_lcomplete_match = 0.0

    dev_ucorrect_nopunc = 0.0
    dev_lcorrect_nopunc = 0.0
    dev_ucomlpete_match_nopunc = 0.0
    dev_lcomplete_match_nopunc = 0.0
    dev_root_correct = 0.0

    best_epoch = 0

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_ucomlpete_match = 0.0
    test_lcomplete_match = 0.0

    test_ucorrect_nopunc = 0.0
    test_lcorrect_nopunc = 0.0
    test_ucomlpete_match_nopunc = 0.0
    test_lcomplete_match_nopunc = 0.0
    test_root_correct = 0.0
    test_total = 0
    test_total_nopunc = 0
    test_total_inst = 0
    test_total_root = 0

    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s, optim: %s, learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, mode, opt, lr, decay_rate, schedule))
        train_err_arc_leaf = 0.
        train_err_arc_non_leaf = 0.
        train_err_type_leaf = 0.
        train_err_type_non_leaf = 0.
        train_err_cov = 0.
        train_total_leaf = 0.
        train_total_non_leaf = 0.
        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
            input_encoder, input_decoder = conllx_stacked_data.get_batch_stacked_variable(data_train, batch_size,
                                                                                          unk_replace=unk_replace)
            word, char, pos, heads, types, masks_e, lengths_e = input_encoder
            stacked_heads, children, stacked_types, masks_d, lengths_d = input_decoder
            optim.zero_grad()
            loss_arc_leaf, loss_arc_non_leaf, \
            loss_type_leaf, loss_type_non_leaf, \
            loss_cov, \
            num_leaf, num_non_leaf = network.loss(word, char, pos, stacked_heads, children, stacked_types,
                                                  mask_e=masks_e, length_e=lengths_e, mask_d=masks_d,
                                                  length_d=lengths_d)
            loss_arc = loss_arc_leaf + loss_arc_non_leaf
            loss_type = loss_type_leaf + loss_type_non_leaf
            loss = loss_arc + loss_type + eta * loss_cov
            loss.backward()
            optim.step()

            num_leaf = num_leaf.data[0]
            num_non_leaf = num_non_leaf.data[0]

            train_err_arc_leaf += loss_arc_leaf.data[0] * num_leaf
            train_err_arc_non_leaf += loss_arc_non_leaf.data[0] * num_non_leaf

            train_err_type_leaf += loss_type_leaf.data[0] * num_leaf
            train_err_type_non_leaf += loss_type_non_leaf.data[0] * num_non_leaf

            train_err_cov += loss_cov.data[0] * (num_leaf + num_non_leaf)

            train_total_leaf += num_leaf
            train_total_non_leaf += num_non_leaf

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 10 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                err_arc_leaf = train_err_arc_leaf / train_total_leaf
                err_arc_non_leaf = train_err_arc_non_leaf / train_total_non_leaf
                err_arc = err_arc_leaf + err_arc_non_leaf

                err_type_leaf = train_err_type_leaf / train_total_leaf
                err_type_non_leaf = train_err_type_non_leaf / train_total_non_leaf
                err_type = err_type_leaf + err_type_non_leaf

                err_cov = train_err_cov / (train_total_leaf + train_total_non_leaf)

                err = err_arc + err_type + eta * err_cov
                log_info = 'train: %d/%d loss (leaf, non_leaf): %.4f, arc: %.4f (%.4f, %.4f), ' \
                           'type: %.4f (%.4f, %.4f), coverage: %.4f, time left (estimated): %.2fs' % (
                               batch, num_batches, err, err_arc, err_arc_leaf, err_arc_non_leaf,
                               err_type, err_type_leaf, err_type_non_leaf, err_cov, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        err_arc_leaf = train_err_arc_leaf / train_total_leaf
        err_arc_non_leaf = train_err_arc_non_leaf / train_total_non_leaf
        err_arc = err_arc_leaf + err_arc_non_leaf

        err_type_leaf = train_err_type_leaf / train_total_leaf
        err_type_non_leaf = train_err_type_non_leaf / train_total_non_leaf
        err_type = err_type_leaf + err_type_non_leaf

        err_cov = train_err_cov / (train_total_leaf + train_total_non_leaf)

        err = err_arc + err_type + eta * err_cov
        print('train: %d loss (leaf, non_leaf): %.4f, arc: %.4f (%.4f, %.4f), type: %.4f (%.4f, %.4f), coverage: %.4f, '
              'time: %.2fs' % (
            num_batches, err, err_arc, err_arc_leaf, err_arc_non_leaf,
            err_type, err_type_leaf, err_type_non_leaf,
            err_cov, time.time() - start_time))

        # evaluate performance on dev data
        network.eval()
        pred_filename = 'tmp/%spred_dev%d' % (str(uid), epoch)
        pred_writer.start(pred_filename)
        gold_filename = 'tmp/%sgold_dev%d' % (str(uid), epoch)
        gold_writer.start(gold_filename)

        dev_ucorr = 0.0
        dev_lcorr = 0.0
        dev_total = 0
        dev_ucomlpete = 0.0
        dev_lcomplete = 0.0
        dev_ucorr_nopunc = 0.0
        dev_lcorr_nopunc = 0.0
        dev_total_nopunc = 0
        dev_ucomlpete_nopunc = 0.0
        dev_lcomplete_nopunc = 0.0
        dev_root_corr = 0.0
        dev_total_root = 0.0
        dev_total_inst = 0.0
        for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_dev, batch_size):
            input_encoder, input_decoder = batch
            word, char, pos, heads, types, masks, lengths = input_encoder
            heads_pred, types_pred = network.decode(word, char, pos, mask=masks, length=lengths, beam=beam)

            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()
            heads = heads.data.cpu().numpy()
            types = types.data.cpu().numpy()

            pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
            gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

            stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types,
                                                                    word_alphabet, pos_alphabet, lengths,
                                                                    punct_set=punct_set, symbolic_root=True)
            ucorr, lcorr, total, ucm, lcm = stats
            ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
            corr_root, total_root = stats_root

            dev_ucorr += ucorr
            dev_lcorr += lcorr
            dev_total += total
            dev_ucomlpete += ucm
            dev_lcomplete += lcm

            dev_ucorr_nopunc += ucorr_nopunc
            dev_lcorr_nopunc += lcorr_nopunc
            dev_total_nopunc += total_nopunc
            dev_ucomlpete_nopunc += ucm_nopunc
            dev_lcomplete_nopunc += lcm_nopunc

            dev_root_corr += corr_root
            dev_total_root += total_root

            dev_total_inst += num_inst

        pred_writer.close()
        gold_writer.close()
        print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
            dev_ucorr, dev_lcorr, dev_total, dev_ucorr * 100 / dev_total, dev_lcorr * 100 / dev_total,
            dev_ucomlpete * 100 / dev_total_inst, dev_lcomplete * 100 / dev_total_inst))
        print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
            dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc, dev_ucorr_nopunc * 100 / dev_total_nopunc,
            dev_lcorr_nopunc * 100 / dev_total_nopunc,
            dev_ucomlpete_nopunc * 100 / dev_total_inst, dev_lcomplete_nopunc * 100 / dev_total_inst))
        print('Root: corr: %d, total: %d, acc: %.2f%%' %(
            dev_root_corr, dev_total_root, dev_root_corr * 100 / dev_total_root))

        if dev_ucorrect_nopunc <= dev_ucorr_nopunc:
            dev_ucorrect_nopunc = dev_ucorr_nopunc
            dev_lcorrect_nopunc = dev_lcorr_nopunc
            dev_ucomlpete_match_nopunc = dev_ucomlpete_nopunc
            dev_lcomplete_match_nopunc = dev_lcomplete_nopunc

            dev_ucorrect = dev_ucorr
            dev_lcorrect = dev_lcorr
            dev_ucomlpete_match = dev_ucomlpete
            dev_lcomplete_match = dev_lcomplete

            dev_root_correct = dev_root_corr

            best_epoch = epoch

            pred_filename = 'tmp/%spred_test%d' % (str(uid), epoch)
            pred_writer.start(pred_filename)
            gold_filename = 'tmp/%sgold_test%d' % (str(uid), epoch)
            gold_writer.start(gold_filename)

            test_ucorrect = 0.0
            test_lcorrect = 0.0
            test_ucomlpete_match = 0.0
            test_lcomplete_match = 0.0
            test_total = 0

            test_ucorrect_nopunc = 0.0
            test_lcorrect_nopunc = 0.0
            test_ucomlpete_match_nopunc = 0.0
            test_lcomplete_match_nopunc = 0.0
            test_total_nopunc = 0
            test_total_inst = 0

            test_root_correct = 0.0
            test_total_root = 0
            for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test, batch_size):
                input_encoder, _ = batch
                word, char, pos, heads, types, masks, lengths = input_encoder
                heads_pred, types_pred = network.decode(word, char, pos, mask=masks, length=lengths, beam=beam)

                word = word.data.cpu().numpy()
                pos = pos.data.cpu().numpy()
                lengths = lengths.cpu().numpy()
                heads = heads.data.cpu().numpy()
                types = types.data.cpu().numpy()

                pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
                gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

                stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types,
                                                                        word_alphabet, pos_alphabet, lengths,
                                                                        punct_set=punct_set, symbolic_root=True)
                ucorr, lcorr, total, ucm, lcm = stats
                ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
                corr_root, total_root = stats_root

                test_ucorrect += ucorr
                test_lcorrect += lcorr
                test_total += total
                test_ucomlpete_match += ucm
                test_lcomplete_match += lcm

                test_ucorrect_nopunc += ucorr_nopunc
                test_lcorrect_nopunc += lcorr_nopunc
                test_total_nopunc += total_nopunc
                test_ucomlpete_match_nopunc += ucm_nopunc
                test_lcomplete_match_nopunc += lcm_nopunc

                test_root_correct += corr_root
                test_total_root += total_root

                test_total_inst += num_inst

            pred_writer.close()
            gold_writer.close()

        print('----------------------------------------------------------------------------------------------------------------------------')
        print('best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            dev_ucorrect, dev_lcorrect, dev_total, dev_ucorrect * 100 / dev_total, dev_lcorrect * 100 / dev_total,
            dev_ucomlpete_match * 100 / dev_total_inst, dev_lcomplete_match * 100 / dev_total_inst,
            best_epoch))
        print('best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            dev_ucorrect_nopunc, dev_lcorrect_nopunc, dev_total_nopunc,
            dev_ucorrect_nopunc * 100 / dev_total_nopunc, dev_lcorrect_nopunc * 100 / dev_total_nopunc,
            dev_ucomlpete_match_nopunc * 100 / dev_total_inst, dev_lcomplete_match_nopunc * 100 / dev_total_inst,
            best_epoch))
        print('best dev  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            dev_root_correct, dev_total_root, dev_root_correct * 100 / dev_total_root, best_epoch))
        print('----------------------------------------------------------------------------------------------------------------------------')
        print('best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total, test_lcorrect * 100 / test_total,
            test_ucomlpete_match * 100 / test_total_inst, test_lcomplete_match * 100 / test_total_inst,
            best_epoch))
        print('best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
            test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
            test_ucomlpete_match_nopunc * 100 / test_total_inst, test_lcomplete_match_nopunc * 100 / test_total_inst,
            best_epoch))
        print('best test Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            test_root_correct, test_total_root, test_root_correct * 100 / test_total_root, best_epoch))
        print('============================================================================================================================')

        if epoch % schedule == 0:
            # lr = lr * decay_rate
            if epoch < adam_epochs:
                opt = 'adam'
                lr = adam_rate / (1.0 + epoch * decay_rate)
                optim = Adam(network.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=gamma)
            else:
                opt = 'sgd'
                lr = learning_rate / (1.0 + (epoch - adam_epochs) * decay_rate)
                optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)


if __name__ == '__main__':
    main()
