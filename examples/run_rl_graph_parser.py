from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os, math

reload(sys)
sys.setdefaultencoding('utf-8')   # Try setting the system default encoding as utf-8 at the start of the script, so that all strings are encoded using that. Or there will be UnicodeDecodeError: 'ascii' codec can't decode byte...

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD, Adamax
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvBiAffine
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding
from seq2seq_rl.seq2seq import Seq2seq_Model
from seq2seq_rl.rl import LossRL, LossBiafRL, get_bleu, get_correct
from stack_parser_eval import third_party_parser
import pickle
from bist_parser.barchybrid.src.arc_hybrid import ArcHybridLSTM

uid = uuid.uuid4().hex[:6]

# 3 sub-models should be pretrained in our approach
#   seq2seq pretrain, denoising autoencoder  | or using token-wise adv to generate adv examples.
#   structure prediction model
#   oracle parser
# then we train the seq2seq model using rl


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU', 'FastLSTM'], help='architecture of rnn', required=True)
    args_parser.add_argument('--cuda', action='store_true', help='using GPU')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--type_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    args_parser.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_parser.add_argument('--pos', action='store_true', help='use part-of-speech embedding.')
    args_parser.add_argument('--char', action='store_true', help='use character embedding and CNN.')
    args_parser.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_parser.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm')
    args_parser.add_argument('--objective', choices=['cross_entropy', 'crf'], default='cross_entropy', help='objective function of training procedure.')
    args_parser.add_argument('--decode', choices=['mst', 'greedy'], help='decoding algorithm', required=True)
    args_parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters', required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)

    args_parser.add_argument('--seq2seq_save_path', default='models/seq2seq/seq2seq_save_model', type=str, help='seq2seq_save_path')
    args_parser.add_argument('--network_save_path', default='models/seq2seq/network_save_model', type=str, help='network_save_path')

    args_parser.add_argument('--seq2seq_load_path', default='models/seq2seq/seq2seq_save_model', type=str, help='seq2seq_load_path')
    args_parser.add_argument('--network_load_path', default='models/seq2seq/network_save_model', type=str, help='network_load_path')

    args_parser.add_argument('--rl_finetune_seq2seq_save_path', default='models/rl_finetune/seq2seq_save_model', type=str, help='rl_finetune_seq2seq_save_path')
    args_parser.add_argument('--rl_finetune_network_save_path', default='models/rl_finetune/network_save_model', type=str, help='rl_finetune_network_save_path')

    args_parser.add_argument('--rl_finetune_seq2seq_load_path', default='models/rl_finetune/seq2seq_save_model', type=str, help='rl_finetune_seq2seq_load_path')
    args_parser.add_argument('--rl_finetune_network_load_path', default='models/rl_finetune/network_save_model', type=str, help='rl_finetune_network_load_path')

    args_parser.add_argument('--treebank', type=str, default='ctb', help='tree bank', choices=['ctb', 'ptb'])  # ctb

    args_parser.add_argument('--direct_eval', action='store_true', help='direct eval without generation process')
    args = args_parser.parse_args()

    # args.train = "data/ptb/dev.conllu"
    # args.dev = "data/ptb/dev.conllu"
    # args.test = "data/ptb/dev.conllu"

    logger = get_logger("GraphParser")

    # SEED = 0
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    mode = args.mode
    obj = args.objective
    decoding = args.decode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    model_path = args.model_path
    model_name = args.model_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    num_layers = args.num_layers
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    opt = args.opt
    momentum = 0.9
    betas = (0.9, 0.9)
    eps = args.epsilon
    decay_rate = args.decay_rate
    clip = args.clip
    gamma = args.gamma
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    punctuation = args.punctuation

    freeze = args.freeze
    word_embedding = args.word_embedding
    word_path = args.word_path

    use_char = args.char
    char_embedding = args.char_embedding
    char_path = args.char_path

    use_pos = args.pos
    pos_dim = args.pos_dim
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)

    char_dict = None
    char_dim = args.char_dim
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path, data_paths=[dev_path, test_path],
                                                                                             max_vocabulary_size=100000, embedd_dict=word_dict)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("Reading Data")
    device = torch.device('cuda')  #torch.device('cuda:0') if args.cuda else torch.device('cpu') #TODO:8.8

    data_train = conllx_data.read_data_to_tensor(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True, device=device)
    # data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    # num_data = sum([len(bucket) for bucket in data_train])
    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data_to_tensor(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True, device=device)
    data_test = conllx_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True, device=device)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
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

    # Pretrain structure prediction model (biaff model). model name: network
    window = 3
    if obj == 'cross_entropy':
        network = BiRecurrentConvBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                                          mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                          embedd_word=word_table, embedd_char=char_table,
                                          p_in=p_in, p_out=p_out, p_rnn=p_rnn, biaffine=True, pos=use_pos, char=use_char)
    elif obj == 'crf':
        raise NotImplementedError
    else:
        raise RuntimeError('Unknown objective: %s' % obj)

    def save_args():
        arg_path = model_name + '.arg.json'
        arguments = [word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                     mode, hidden_size, num_layers, num_types, arc_space, type_space]
        kwargs = {'p_in': p_in, 'p_out': p_out, 'p_rnn': p_rnn, 'biaffine': True, 'pos': use_pos, 'char': use_char}
        json.dump({'args': arguments, 'kwargs': kwargs}, open(arg_path, 'w'), indent=4)

    if freeze:
        freeze_embedding(network.word_embedd)

    network = network.to(device)

    save_args()

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    def generate_optimizer(opt, lr, params):
        params = filter(lambda param: param.requires_grad, params)
        if opt == 'adam':
            return Adam(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)
        elif opt == 'sgd':
            return SGD(params, lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
        elif opt == 'adamax':
            return Adamax(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)
        else:
            raise ValueError('Unknown optimization algorithm: %s' % opt)

    lr = learning_rate
    optim = generate_optimizer(opt, lr, network.parameters())
    opt_info = 'opt: %s, ' % opt
    if opt == 'adam':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)
    elif opt == 'sgd':
        opt_info += 'momentum=%.2f' % momentum
    elif opt == 'adamax':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)

    word_status = 'frozen' if freeze else 'fine tune'
    char_status = 'enabled' if use_char else 'disabled'
    pos_status = 'enabled' if use_pos else 'disabled'
    logger.info("Embedding dim: word=%d (%s), char=%d (%s), pos=%d (%s)" % (word_dim, word_status, char_dim, char_status, pos_dim, pos_status))
    logger.info("CNN: filter=%d, kernel=%d" % (num_filters, window))
    logger.info("RNN: %s, num_layer=%d, hidden=%d, arc_space=%d, type_space=%d" % (mode, num_layers, hidden_size, arc_space, type_space))
    logger.info("train: obj: %s, l2: %f, (#data: %d, batch: %d, clip: %.2f, unk replace: %.2f)" % (obj, gamma, num_data, batch_size, clip, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))
    logger.info("decoding algorithm: %s" % decoding)
    logger.info(opt_info)

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

    if decoding == 'greedy':
        decode = network.decode
    elif decoding == 'mst':
        decode = network.decode_mst
    else:
        raise ValueError('Unknown decoding algorithm: %s' % decoding)

    print('Pretrain biaffine model.')
    patient = 0
    decay = 0
    max_decay = 9
    double_schedule_decay = 5
    num_epochs = 0  # debug hanwj
    if args.treebank == 'ptb':
        network.load_state_dict(torch.load('models/parsing/biaffine/network.pt'))  # TODO: 10.7
    elif args.treebank == 'ctb':
        network.load_state_dict(torch.load('ctb_models/parsing/biaffine/network.pt'))  # TODO: 10.7
    network.to(device)
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s, optim: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f (schedule=%d, patient=%d, decay=%d)): ' % (epoch, mode, opt, lr, eps, decay_rate, schedule, patient, decay))
        train_err = 0.
        train_err_arc = 0.
        train_err_type = 0.
        train_total = 0.
        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
            word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_train, batch_size, unk_replace=unk_replace)

            optim.zero_grad()
            loss_arc, loss_type = network.loss(word, char, pos, heads, types, mask=masks, length=lengths)
            loss = loss_arc + loss_type
            loss.backward()
            clip_grad_norm_(network.parameters(), clip)
            optim.step()

            with torch.no_grad():
                num_inst = word.size(0) if obj == 'crf' else masks.sum() - word.size(0)
                train_err += loss * num_inst
                train_err_arc += loss_arc * num_inst
                train_err_type += loss_type * num_inst
                train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 10 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, arc: %.4f, type: %.4f, time left: %.2fs' % (batch, num_batches, train_err / train_total,
                                                                                                 train_err_arc / train_total, train_err_type / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, arc: %.4f, type: %.4f, time: %.2fs' % (num_batches, train_err / train_total,
                                                                            train_err_arc / train_total, train_err_type / train_total, time.time() - start_time))

        # evaluate performance on dev data
        with torch.no_grad():
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
            for batch in conllx_data.iterate_batch_tensor(data_dev, batch_size):
                word, char, pos, heads, types, masks, lengths = batch
                heads_pred, types_pred = decode(word, char, pos, mask=masks, length=lengths, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                word = word.cpu().numpy()
                pos = pos.cpu().numpy()
                lengths = lengths.cpu().numpy()
                heads = heads.cpu().numpy()
                types = types.cpu().numpy()

                pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
                gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

                stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types,
                                                                        word_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
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
            print('Root: corr: %d, total: %d, acc: %.2f%%' %(dev_root_corr, dev_total_root, dev_root_corr * 100 / dev_total_root))

            if dev_lcorrect_nopunc < dev_lcorr_nopunc or (dev_lcorrect_nopunc == dev_lcorr_nopunc and dev_ucorrect_nopunc < dev_ucorr_nopunc):
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
                patient = 0
                # torch.save(network, model_name)
                torch.save(network.state_dict(), model_name)

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
                for batch in conllx_data.iterate_batch_tensor(data_test, batch_size):
                    word, char, pos, heads, types, masks, lengths = batch
                    heads_pred, types_pred = decode(word, char, pos, mask=masks, length=lengths, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    word = word.cpu().numpy()
                    pos = pos.cpu().numpy()
                    lengths = lengths.cpu().numpy()
                    heads = heads.cpu().numpy()
                    types = types.cpu().numpy()

                    pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
                    gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

                    stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types,
                                                                            word_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
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
            else:
                if dev_ucorr_nopunc * 100 / dev_total_nopunc < dev_ucorrect_nopunc * 100 / dev_total_nopunc - 5 or patient >= schedule:
                    # network = torch.load(model_name)
                    network.load_state_dict(torch.load(model_name))
                    lr = lr * decay_rate
                    optim = generate_optimizer(opt, lr, network.parameters())

                    if decoding == 'greedy':
                        decode = network.decode
                    elif decoding == 'mst':
                        decode = network.decode_mst
                    else:
                        raise ValueError('Unknown decoding algorithm: %s' % decoding)

                    patient = 0
                    decay += 1
                    if decay % double_schedule_decay == 0:
                        schedule *= 2
                else:
                    patient += 1

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

            if decay == max_decay:
                break

    # Pretrain seq2seq model using denoising autoencoder. model name: seq2seq model
    print('Pretrain seq2seq model using denoising autoencoder.')
    EPOCHS = 0  # 150
    DECAY = 0.97
    shared_word_embedd = network.return_word_embedd()
    shared_word_embedd.weight.requires_grad = False
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=shared_word_embedd, device=device).to(device)  # debug hanwj
    seq2seq.emb.weight.requires_grad = False
    print(seq2seq)

    loss_seq2seq = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_seq2seq = torch.optim.Adam(parameters_need_update, lr=0.0002)
    if args.treebank == 'ptb':
        seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(2) + '.pt'))  # TODO: 10.7
        seq2seq.to(device)
        network.load_state_dict(torch.load(args.network_load_path + str(2) + '.pt'))  # TODO: 10.7
        network.to(device)
    elif args.treebank == 'ctb':
        seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(7) + '.pt'))  # TODO: 10.7
        seq2seq.to(device)
        network.load_state_dict(torch.load(args.network_load_path + str(7) + '.pt'))  # TODO: 10.7
        network.to(device)

    for i in range(EPOCHS):
        ls_seq2seq_ep = 0
        seq2seq.train()
        network.train()
        seq2seq.emb.weight.requires_grad = False
        print('----------'+str(i)+' iter----------')
        for _ in range(1, num_batches + 1):
            word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_train, batch_size,
                                                                                         unk_replace=unk_replace)  # word:(32,50)  char:(32,50,35)
            inp, _ = seq2seq.add_noise(word, lengths)
            dec_out = word
            dec_inp = torch.cat((word[:,0:1], word[:,0:-1]), dim=1)  # maybe wrong
            # train_seq2seq
            out = seq2seq(inp.long().to(device), is_tr=True, dec_inp=dec_inp.long().to(device))

            out = out.view((out.shape[0] * out.shape[1], out.shape[2]))
            dec_out = dec_out.view((dec_out.shape[0] * dec_out.shape[1],))
            wgt = seq2seq.add_stop_token(masks, lengths)
            wgt = wgt.view((wgt.shape[0] * wgt.shape[1],)).float().to(device)

            ls_seq2seq_bh = loss_seq2seq(out, dec_out.long().to(device))
            ls_seq2seq_bh = (ls_seq2seq_bh * wgt).sum() / wgt.sum()

            optim_seq2seq.zero_grad()
            ls_seq2seq_bh.backward()
            optim_seq2seq.step()

            ls_seq2seq_bh = ls_seq2seq_bh.cpu().detach().numpy()
            ls_seq2seq_ep += ls_seq2seq_bh
        print('ls_seq2seq_ep: ', ls_seq2seq_ep)
        for pg in optim_seq2seq.param_groups:
            pg['lr'] *= DECAY

        # test th bleu of seq2seq
        if False: #i%1 == 0:
            seq2seq.eval()
            network.eval()
            bleu_ep = 0
            acc_numerator_ep = 0
            acc_denominator_ep = 0
            testi = 0
            for batch in conllx_data.iterate_batch_tensor(data_dev, batch_size):   # for _ in range(1, num_batches + 1):  word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_dev, batch_size, unk_replace=unk_replace)  # word:(32,50)  char:(32,50,35)
                word, char, pos, heads, types, masks, lengths = batch
                inp = word
                # inp, _ = seq2seq.add_noise(word, lengths)
                dec_out = word
                sel, _ = seq2seq(inp.long().to(device), LEN=inp.size()[1])
                sel = sel.detach().cpu().numpy()
                dec_out = dec_out.cpu().numpy()

                bleus = []
                for j in range(sel.shape[0]):
                    bleu = get_bleu(sel[j], dec_out[j], num_words)  # sel
                    bleus.append(bleu)
                    numerator, denominator = get_correct(sel[j], dec_out[j], num_words)
                    acc_numerator_ep += numerator
                    acc_denominator_ep += denominator #.detach().cpu().numpy() TODO: 10.8
                bleu_bh = np.average(bleus)
                bleu_ep += bleu_bh
                testi += 1
            bleu_ep /= testi  #num_batches
            print('testi: ', testi)
            print('Valid bleu: %.4f%%' % (bleu_ep * 100))
            # print(acc_denominator_ep)
            print('Valid acc: %.4f%%' % ((acc_numerator_ep*1.0/acc_denominator_ep) * 100))
        # for debug TODO:
        if i > 0:
            torch.save(seq2seq.state_dict(), args.seq2seq_save_path + str(i) + '.pt')
            torch.save(network.state_dict(), args.network_save_path + str(i) + '.pt')
    # Pretrain seq2seq model using token wise adv examples. model name: seq2seq model
    # print('Pretrain seq2seq model using token wise adv examples.')

    # Train seq2seq model using rl with reward of biaffine. model name: seq2seq model
    print('Train seq2seq model using rl with reward of biaffine.')

    # import third_party_parser
    sudo_golden_parser = third_party_parser(device, word_table, char_table, 0, args)
    sudo_golden_parser_1 = third_party_parser(device, word_table, char_table, 1, args)
    sudo_golden_parser.eval()
    sudo_golden_parser_1.eval()


    if args.treebank == 'ptb':
        params = 'bist_parser/pretrained/model1/params.pickle'
        external_embedding = 'bist_parser/sskip.100.vectors'
        model = 'bist_parser/pretrained/model1/barchybrid.model30'
    elif args.treebank == 'ctb':
        params = 'bist_parser/ctb_output/params.pickle'
        external_embedding = 'bist_parser/sskip.chn.50'
        model = 'bist_parser/ctb_output/barchybrid.model30'
    with open(params, 'r') as paramsfp:
        words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

    stored_opt.external_embedding = external_embedding
    bist_parser = ArcHybridLSTM(words, pos, rels, w2i, stored_opt)
    bist_parser.Load(model)

    EPOCHS = 80 #0  # 80
    DECAY = 0.97
    M = 1  # this is the size of beam searching ?
    seq2seq.emb.weight.requires_grad = False
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_bia_rl = torch.optim.Adam(parameters_need_update, lr=1e-5)  #1e-5 0.00005
    loss_biaf_rl = LossBiafRL(device=device, word_alphabet=word_alphabet, vocab_size=num_words).to(device)

    seq2seq.load_state_dict(torch.load(args.rl_finetune_seq2seq_load_path + str(4) + '.pt'))  # TODO: 7.13
    seq2seq.to(device)
    network.load_state_dict(torch.load(args.rl_finetune_network_load_path + str(4) + '.pt'))  # TODO: 7.13
    network.to(device)

    parser_select = ['stackPtr0', 'bist']

    for epoch_i in range(EPOCHS):
        print('======='+str(epoch_i)+'=========')
        ls_rl_ep = rewards1 = rewards2 = rewards3 = rewards4 = rewards5 = 0
        network.eval()  # only train seq2seq
        seq2seq.train()
        seq2seq.emb.weight.requires_grad = False
        END_token = word_alphabet.instance2index['_PAD']  # word_alphabet.get_instance('_PAD)==1  '_END'==3
        # num_batches = 100   # TODO:8.9
        if args.treebank == 'ptb':
            batch_size = 10
        elif args.treebank == 'ctb':
            batch_size = 1
        num_batches = 0
        print('num_batches: ', str(num_batches))
        for kkk in range(1, num_batches + 1): #num_batches
            # print('---'+str(kkk)+'---')
            # train_rl
            word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_train, batch_size, unk_replace=unk_replace)
            inp = word
            if True:  #inp.size()[1]<15:#True:  #inp.size()[1]<15: #TODO: debug hanwj
                decode = network.decode_mst
                _, sel, pb = seq2seq(inp.long().to(device), is_tr=True, M=M, LEN=inp.size()[1])
                sel1 = sel.data.detach()
                try:
                    end_position = torch.eq(sel1, END_token).nonzero()
                except RuntimeError:
                    continue
                    print(sel1)
                masks_sel = torch.ones_like(sel1, dtype=torch.float)
                lengths_sel = torch.ones_like(lengths).fill_(sel1.shape[1])  #sel1.shape[1]-1 TODO: because of end token in the end
                if not len(end_position)==0:
                    ij_back = -1
                    for ij in end_position:
                        if not (ij[0]==ij_back):
                            lengths_sel[ij[0]] = ij[1]
                            masks_sel[ij[0], ij[1]:] = 0  # -1 TODO: because of end token in the end
                            ij_back = ij[0]

                with torch.no_grad():
                    try:
                        heads_pred, types_pred = decode(sel1, input_char=None, input_pos=None, mask=masks_sel, length=lengths_sel,
                                                        leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    except:
                        print('IndexError Maybe sel1: ', sel1)
                        print('IndexError Maybe: ', sel1.data.cpu().numpy())
                        print(masks_sel)
                        continue
                    if 'stackPtr0' in parser_select:
                        sudo_heads_pred, sudo_types_pred = sudo_golden_parser.parsing(sel1, None, None, masks_sel, lengths_sel,
                                                                                  beam=1)  # beam=1 ?? it should be equal to M TODO:
                    if 'stackPtr1' in parser_select:
                        sudo_heads_pred_1, sudo_types_pred_1 = sudo_golden_parser_1.parsing(sel1, None, None, masks_sel, lengths_sel,
                                                                                        beam=1)  # beam=1 ?? it should be equal to M TODO:
                    elif 'bist' in parser_select:
                        str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in one_stc] for one_stc in sel1.cpu().numpy()]
                        stc_pred_1 = list(bist_parser.predict_stcs(str_sel, lengths_sel))
                        sudo_heads_pred_1 = np.array([[one_w.pred_parent_id for one_w in stc]+[0 for _ in range(sel1.shape[1]-len(stc))] for stc in stc_pred_1])
                ls_rl_bh, reward1, reward2, reward3, reward4, reward5 = loss_biaf_rl(sel, pb, predicted_out=heads_pred, golden_out=heads, mask_id=END_token,
                                                                              stc_length_out=lengths_sel, sudo_golden_out=sudo_heads_pred, sudo_golden_out_1=sudo_heads_pred_1,
                                                                              ori_words=word, ori_words_length=lengths)  #sudo_heads_pred_1 TODO: (sel, pb, heads)  # heads is replaced by dec_out.long().to(device)
                optim_bia_rl.zero_grad()
                ls_rl_bh.backward()
                optim_bia_rl.step()
                ls_rl_bh = ls_rl_bh.cpu().detach().numpy()
                ls_rl_ep += ls_rl_bh
                # ls1 = ls1.cpu().detach().numpy()
                rewards1 += reward1
                rewards2 += reward2
                rewards3 += reward3
                rewards4 += reward4
                rewards5 += reward5
        if False:
            print('train loss: ', ls_rl_ep)
            print('train reward parser b: ', rewards1)
            print('train reward parser c: ', rewards2)
            print('train reward parser b^c: ', rewards3)
            print('train reward meaning: ', rewards4)
            print('train reward fluency: ', rewards5)
        for pg in optim_bia_rl.param_groups:
            pg['lr'] *= DECAY

        if epoch_i > 0:
            torch.save(seq2seq.state_dict(), args.rl_finetune_seq2seq_save_path + str(epoch_i) + '.pt')
            torch.save(network.state_dict(), args.rl_finetune_network_save_path + str(epoch_i) + '.pt')

        ####eval######
        seq2seq.eval()
        network.eval()
        ls_rl_ep = rewards1 = rewards2 = rewards3 = rewards4 = rewards5 = rewardsall1 = rewardsall2 = rewardsall3 = 0
        pred_writer_test = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        if args.treebank == 'ptb':
            pred_filename_test = 'dumped/pred_test%d' % (epoch_i)
            src_filename_test = 'dumped/src_test%d' % (epoch_i)
        elif args.treebank == 'ctb':
            src_filename_test = 'ctb_dumped/src_test%d' % (epoch_i)
            pred_filename_test = 'ctb_dumped/pred_test%d' % (epoch_i)

        pred_writer_test.start(pred_filename_test)

        src_writer_test = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        src_writer_test.start(src_filename_test)

        pred_parse_writer_testA = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        pred_parse_writer_testA.start(pred_filename_test+'_parseA.txt')

        pred_parse_writer_testB = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        pred_parse_writer_testB.start(pred_filename_test+'_parseB.txt')

        pred_parse_writer_testC = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
        pred_parse_writer_testC.start(pred_filename_test+'_parseC.txt')

        nll = 0
        token_num = 0
        kk = 0
        batch_size_for_eval = 20
        for batch in conllx_data.iterate_batch_tensor(data_test, batch_size_for_eval):  # batch_size
            kk += 1
            print('-------'+str(kk)+'-------')
            if kk > 100:  # TODO:8.9
                break
            word, char, pos, heads, types, masks, lengths = batch
            # print(lengths)
            print(args.direct_eval)
            if not args.direct_eval:
                inp = word  #, _ = seq2seq.add_noise(word, lengths)
                sel, pb = seq2seq(inp.long().to(device), LEN=inp.size()[1])
                end_position = torch.eq(sel, END_token).nonzero()
                masks_sel = torch.ones_like(sel, dtype=torch.float)
                lengths_sel = torch.ones_like(lengths).fill_(sel.shape[1])  # sel1.shape[1]-1 TODO: because of end token in the end
                if not len(end_position) == 0:
                    ij_back = -1
                    for ij in end_position:
                        if not (ij[0]==ij_back):
                            lengths_sel[ij[0]] = ij[1]
                            masks_sel[ij[0], ij[1]:] = 0  # -1 TODO: because of end token in the end
                            ij_back = ij[0]
            else:
                sel = word
                pb = torch.ones_like(sel, dtype=torch.float).fill_(0)
                lengths_sel = lengths
                masks_sel = masks
            with torch.no_grad():
                heads_pred, types_pred = decode(sel, input_char=None, input_pos=None, mask=masks_sel,
                                                length=lengths_sel,
                                                leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                if 'stackPtr0' in parser_select:
                    sudo_heads_pred, sudo_types_pred = sudo_golden_parser.parsing(sel, None, None, masks_sel,
                                                                                  lengths_sel,
                                                                                  beam=1)  # beam=1 ?? it should be equal to M TODO:
                if 'stackPtr1' in parser_select:
                    sudo_heads_pred_1, sudo_types_pred_1 = sudo_golden_parser_1.parsing(sel, None, None, masks_sel,
                                                                                        lengths_sel,
                                                                                        beam=1)  # beam=1 ?? it should be equal to M TODO:
                elif 'bist' in parser_select:
                    str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in one_stc] for
                               one_stc in sel.cpu().numpy()]
                    stc_pred_1 = list(bist_parser.predict_stcs(str_sel, lengths_sel))
                    sudo_heads_pred_1 = np.array(
                        [[one_w.pred_parent_id for one_w in stc] + [0 for _ in range(sel.shape[1] - len(stc))] for stc
                         in stc_pred_1])
                # ls_rl_bh, reward1, reward2, reward3, reward4, reward5 = loss_biaf_rl(sel, pb, predicted_out=heads_pred,
                #                                                               golden_out=heads, mask_id=END_token,
                #                                                               stc_length_out=lengths_sel,
                #                                                               sudo_golden_out=sudo_heads_pred,
                #                                                               sudo_golden_out_1=sudo_heads_pred_1,
                #                                                               ori_words=word,
                #                                                               ori_words_length=lengths
                #                                                               )  # TODO: (sel, pb, heads)  # heads is replaced by dec_out.long().to(device)
                #

                ls_rl_bh, _ , _ , _ , _ , _ , reward1, reward2, reward3, reward4, reward5 , rewardall1, rewardall2, rewardall3 = loss_biaf_rl.forward_verbose(sel, pb, predicted_out=heads_pred,
                                                                                                                                                              golden_out=heads, mask_id=END_token,
                                                                                                                                                              stc_length_out=lengths_sel,
                                                                                                                                                              sudo_golden_out=sudo_heads_pred,
                                                                                                                                                              sudo_golden_out_1=sudo_heads_pred_1,
                                                                                                                                                              ori_words=word,
                                                                                                                                                              ori_words_length=lengths
                                                                                                                                                              )  # TODO: (sel, pb, heads)  # heads is replaced by dec_out.long().to(device)



            ls_rl_bh = ls_rl_bh.cpu().detach().numpy()
            ls_rl_ep += ls_rl_bh
            rewards1 += reward1
            rewards2 += reward2
            rewards3 += reward3
            rewards4 += reward4
            rewards5 += reward5
            rewardsall1 += rewardall1
            rewardsall2 += rewardall2
            rewardsall3 += rewardall3
            sel = sel.detach().cpu().numpy()
            lengths_sel = lengths_sel.detach().cpu().numpy()
            # print(sel)
            pred_writer_test.write_stc(sel, lengths_sel, symbolic_root=True)
            src_writer_test.write_stc(word, lengths, symbolic_root=True)
            pred_parse_writer_testA.write(sel, sel, heads_pred, types_pred, lengths_sel, symbolic_root=True)  # word, pos, head, type, lengths,
            pred_parse_writer_testB.write(sel, sel, sudo_heads_pred, types_pred, lengths_sel, symbolic_root=True)  # word, pos, head, type, lengths,
            pred_parse_writer_testC.write(sel, sel, sudo_heads_pred_1, types_pred, lengths_sel, symbolic_root=True)  # word, pos, head, type, lengths,

            for i in range(len(lengths_sel)):
                nll += sum(pb[i, 1:lengths_sel[i]])
            token_num += sum(lengths_sel)-len(lengths_sel)

        rewards1 = rewards1 * 1.0 / sum(data_test[1])
        rewards2 = rewards2 * 1.0 / sum(data_test[1])
        rewards3 = rewards3 * 1.0 / sum(data_test[1])
        rewards4 = rewards4 * 1.0 / sum(data_test[1])
        rewards5 = rewards5 * 1.0 / sum(data_test[1])
        rewardsall1 = rewardsall1 * 1.0 / sum(data_test[1])
        rewardsall2 = rewardsall2 * 1.0 / sum(data_test[1])
        rewardsall3 = rewardsall3 * 1.0 / sum(data_test[1])

        nll /= token_num

        print('test loss: ', ls_rl_ep)
        print('test metrics parser b: ', rewards1)
        print('test metrics parser c: ', rewards2)
        print('test metrics parser b^c: ', rewards3)
        print('test metrics meaning: ', rewards4)
        print('test metrics fluency: ', rewards5)
        print('test metrics whole parser b: ', rewardsall1)
        print('test metrics whole parser c: ', rewardsall2)
        print('test metrics whole parser b^c: ', rewardsall3)

        print('test nll: ', nll)

        pred_writer_test.close()
        src_writer_test.close()
        pred_parse_writer_testA.close()
        pred_parse_writer_testB.close()
        pred_parse_writer_testC.close()


if __name__ == '__main__':
    main()
