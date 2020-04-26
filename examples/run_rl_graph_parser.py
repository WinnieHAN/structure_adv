from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.append(".")
sys.path.append("..")

import argparse
import uuid
import json

import numpy as np
import torch
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvBiAffine
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.nn.utils import freeze_embedding
from seq2seq_rl.seq2seq import Seq2seq_Model
from seq2seq_rl.rl import LossBiafRL
from stack_parser_eval import third_party_parser
import pickle
from bist_parser.barchybrid.src.arc_hybrid import ArcHybridLSTM
import global_variables

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
    args_parser.add_argument('--port', type=int, default=10048, help='localhost port for berscore server')
    args_parser.add_argument('--z1_weight', type=float, default=1.0, help='reward weight of z1')
    args_parser.add_argument('--z2_weight', type=float, default=1.0, help='reward weight of z2')
    args_parser.add_argument('--z3_weight', type=float, default=1.0, help='reward weight of z3')
    args_parser.add_argument('--mp_weight', type=float, default=100, help='reward weight of meaning preservation')
    args_parser.add_argument('--ppl_weight', type=float, default=0.001, help='reward weight of ppl')
    args_parser.add_argument('--unk_weight', type=float, default=100, help='reward weight of unk rate')
    args_parser.add_argument('--prefix', type=str, default='/home/zhanglw/code/structure_adv/test_')
    args = args_parser.parse_args()

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

    # optimizer parameter
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
    port = args.port

    freeze = args.freeze
    word_embedding = args.word_embedding
    word_path = args.word_path
    use_char = args.char
    char_embedding = args.char_embedding
    char_path = args.char_path

    # rl weight
    global_variables.Z1_REWARD_WEIGHT = args.z1_weight
    global_variables.Z2_REWARD_WEIGHT = args.z2_weight
    global_variables.Z3_REWARD_WEIGHT = args.z3_weight
    global_variables.MP_REWARD_WEIGHT = args.mp_weight
    global_variables.PPL_REWARD_WEIGHT = args.ppl_weight
    global_variables.UNK_REWARD_WEIGHT = args.unk_weight

    global_variables.PREFIX = args.prefix
    SCORE_PREFIX = args.prefix.split('/')[-1]

    parser_select = ['stackPtr', 'bist']

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
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                             data_paths=[dev_path, test_path],
                                                                                             max_vocabulary_size=100000,
                                                                                             embedd_dict=word_dict)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("Reading Data")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    data_train = conllx_data.read_data_to_tensor(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                 symbolic_root=True, device=device)
    # data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    # num_data = sum([len(bucket) for bucket in data_train])
    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data_to_tensor(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                               symbolic_root=True, device=device)
    data_test = conllx_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                symbolic_root=True, device=device)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(
            -scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale,
                                                                                                        [1,
                                                                                                         word_dim]).astype(
                    np.float32)
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
        network = BiRecurrentConvBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters,
                                          window,
                                          mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                          embedd_word=word_table, embedd_char=char_table,
                                          p_in=p_in, p_out=p_out, p_rnn=p_rnn, biaffine=True, pos=use_pos,
                                          char=use_char)
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

    if args.treebank == 'ptb':
        network.load_state_dict(torch.load('models/parsing/biaffine/network.pt'))
    elif args.treebank == 'ctb':
        network.load_state_dict(torch.load('ctb_models/parsing/biaffine/network.pt'))
    network.to(device)

    shared_word_embedd = network.return_word_embedd()
    shared_word_embedd.weight.requires_grad = False
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words,
                            word_embedd=shared_word_embedd, device=device).to(device)
    seq2seq.emb.weight.requires_grad = False
    print(seq2seq)

    if args.treebank == 'ptb':
        seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(2) + '.pt'))
        seq2seq.to(device)
        network.load_state_dict(torch.load(args.network_load_path + str(2) + '.pt'))
        network.to(device)
    elif args.treebank == 'ctb':
        seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(7) + '.pt'))
        seq2seq.to(device)
        network.load_state_dict(torch.load(args.network_load_path + str(7) + '.pt'))
        network.to(device)

    # import third_party_parser
    if parser_select[0] == 'stackPtr':
        sudo_golden_parser = third_party_parser(device, word_table, char_table, './models/parsing/stack_ptr/')
        sudo_golden_parser.eval()
        bist_parser = None
    elif parser_select[0] == 'bist':
        sudo_golden_parser = None
        if args.treebank == 'ptb':
            params = 'bist_parser/pretrained/model1/params.pickle'
            external_embedding = 'bist_parser/sskip.100.vectors'
            model = 'bist_parser/pretrained/model1/barchybrid.model30'
        elif args.treebank == 'ctb':
            params = 'bist_parser/ctb_output/params.pickle'
            external_embedding = 'bist_parser/sskip.chn.50'
            model = 'bist_parser/ctb_output/barchybrid.model30'
        else:
            raise ValueError('treebank should be ptb or ctb')

        with open(params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = external_embedding
        bist_parser = ArcHybridLSTM(words, pos, rels, w2i, stored_opt)
        bist_parser.Load(model)

    elif parser_select[0] == 'biaffine':
        biaffine_parser = BiRecurrentConvBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters,
                                                  window, mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                                  embedd_word=word_table, embedd_char=char_table,p_in=p_in, p_out=p_out,
                                                  p_rnn=p_rnn, biaffine=True, pos=use_pos, char=use_char)
        if freeze:
            freeze_embedding(biaffine_parser.word_embedd)

        biaffine_parser = biaffine_parser.to(device)
        biaffine_parser.load_state_dict(torch.load('models/parsing/biaffine1/network.pt'))
        biaffine_parser.eval()

    if parser_select[1] == 'stackPtr':
        sudo_golden_parser_1 = third_party_parser(device, word_table, char_table, './models/parsing/stack_ptr1/')
        sudo_golden_parser_1.eval()
        bist_parser_1 = None
    elif parser_select[1] == 'bist':
        sudo_golden_parser_1 = None

        if args.treebank == 'ptb':
            params = 'bist_parser/pretrained/model1/params.pickle'
            external_embedding = 'bist_parser/sskip.100.vectors'
            model = 'bist_parser/pretrained/model1/barchybrid.model30'
        elif args.treebank == 'ctb':
            params = 'bist_parser/ctb_output/params.pickle'
            external_embedding = 'bist_parser/sskip.chn.50'
            model = 'bist_parser/ctb_output/barchybrid.model30'
        else:
            raise ValueError('treebank should be ptb or ctb')

        with open(params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = external_embedding
        bist_parser_1 = ArcHybridLSTM(words, pos, rels, w2i, stored_opt)
        bist_parser_1.Load(model)

    elif parser_select[1] == 'biaffine':
        biaffine_parser_1 = BiRecurrentConvBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters,
                                                  window, mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                                  embedd_word=word_table, embedd_char=char_table,p_in=p_in, p_out=p_out,
                                                  p_rnn=p_rnn, biaffine=True, pos=use_pos, char=use_char)
        if freeze:
            freeze_embedding(biaffine_parser_1.word_embedd)

        biaffine_parser_1 = biaffine_parser_1.to(device)
        biaffine_parser_1.load_state_dict(torch.load('models/parsing/biaffine2/network.pt'))
        biaffine_parser_1.eval()

    M = 1  # this is the size of beam searching ?
    seq2seq.emb.weight.requires_grad = False
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_bia_rl = torch.optim.Adam(parameters_need_update, lr=learning_rate)  # 1e-5 0.00005
    loss_biaf_rl = LossBiafRL(device=device, word_alphabet=word_alphabet, vocab_size=num_words, port=port).to(device)

    # seq2seq.load_state_dict(torch.load(args.rl_finetune_seq2seq_load_path + '_' + SCORE_PREFIX + str(0) + '.pt'))  # TODO: 7.13
    seq2seq.to(device)
    # network.load_state_dict(torch.load(args.rl_finetune_network_load_path + '_' + SCORE_PREFIX + str(0) + '.pt'))  # TODO: 7.13
    network.to(device)

    for epoch_i in range(0, num_epochs):
        print('=======' + str(epoch_i) + '=========')
        ls_rl_ep = rewards1 = rewards2 = rewards3 = rewards4 = rewards5 = rewards6 = 0
        network.eval()  # only train seq2seq
        seq2seq.train()
        seq2seq.emb.weight.requires_grad = False
        END_token = word_alphabet.instance2index['_PAD']  # word_alphabet.get_instance('_PAD)==1  '_END'==3
        train_num = test_num = 0
        # num_batches =
        # if args.treebank == 'ptb':
        #     batch_size = 10
        # elif args.treebank == 'ctb':
        #     batch_size = 10
        # num_batches = 0

        for kkk in range(1, num_batches + 1):  # range(1, 1):
            # print('---'+str(kkk)+'---')
            # train_rl
            word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_train, batch_size, unk_replace=unk_replace)
            inp = word
            # print("Batch " + str(kkk) + " Size: " + str(inp.size()))
            if inp.size()[1] >= 140:
                print("Skip batch not smaller than 140")
                continue
            if True:  # inp.size()[1]<15:#True:  #inp.size()[1]<15:
                decode = network.decode_mst
                _, sel, pb = seq2seq(inp.long().to(device), is_tr=True, M=M, LEN=inp.size()[1])
                sel1 = sel.data.detach()
                try:
                    end_position = torch.eq(sel1, END_token).nonzero()
                except RuntimeError:
                    continue

                masks_sel = torch.ones_like(sel1, dtype=torch.float)
                lengths_sel = torch.ones_like(lengths).fill_(
                    sel1.shape[1])  # sel1.shape[1]-1 TODO: because of end token in the end
                if not len(end_position) == 0:
                    ij_back = -1
                    for ij in end_position:
                        if not (ij[0] == ij_back):
                            lengths_sel[ij[0]] = ij[1]
                            masks_sel[ij[0], ij[1]:] = 0  # -1 TODO: because of end token in the end
                            ij_back = ij[0]

                with torch.no_grad():
                    try:
                        heads_pred, types_pred = decode(sel1, input_char=None, input_pos=None, mask=masks_sel,
                                                        length=lengths_sel,
                                                        leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    except:
                        print('IndexError Maybe sel1: ', sel1)
                        print('IndexError Maybe: ', sel1.data.cpu().numpy())
                        print(masks_sel)
                        continue

                    if 'stackPtr' == parser_select[0]:
                        sudo_heads_pred, sudo_types_pred = sudo_golden_parser.parsing(sel1, None, None, masks_sel,
                                                                                      lengths_sel, beam=1)
                    elif 'bist' == parser_select[0]:
                        str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in one_stc] for
                                   one_stc in sel1.cpu().numpy()]
                        stc_pred = list(bist_parser.predict_stcs(str_sel, lengths_sel))
                        sudo_heads_pred = np.array(
                            [[one_w.pred_parent_id for one_w in stc] + [0 for _ in range(sel1.shape[1] - len(stc))] for
                             stc in stc_pred])
                    elif parser_select[0] == 'biaffine':
                        sudo_heads_pred, _ = biaffine_parser.decode_mst(sel1, input_char=None, input_pos=None, mask=masks_sel,
                                                        length=lengths_sel, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    else:
                        raise ValueError('Error first parser select code!')

                    if 'stackPtr' == parser_select[1]:
                        sudo_heads_pred_1, sudo_types_pred_1 = sudo_golden_parser_1.parsing(sel1, None, None, masks_sel,
                                                                                            lengths_sel, beam=1)
                    elif 'bist' == parser_select[1]:
                        str_sel_1 = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in one_stc] for
                                     one_stc in sel1.cpu().numpy()]
                        stc_pred_1 = list(bist_parser_1.predict_stcs(str_sel_1, lengths_sel))
                        sudo_heads_pred_1 = np.array(
                            [[one_w.pred_parent_id for one_w in stc] + [0 for _ in range(sel1.shape[1] - len(stc))] for
                             stc in stc_pred_1])
                    elif parser_select[1] == 'biaffine':
                        sudo_heads_pred_1, _ = biaffine_parser_1.decode_mst(sel1, input_char=None, input_pos=None, mask=masks_sel,
                                                        length=lengths_sel, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    else:
                        raise ValueError('Error second parser select code!')

                ls_rl_bh, reward1, reward2, reward3, reward4, reward5, reward6 = loss_biaf_rl(sel, pb, predicted_out=heads_pred,
                                                                                     golden_out=heads,
                                                                                     mask_id=END_token,
                                                                                     stc_length_out=lengths_sel,
                                                                                     sudo_golden_out=sudo_heads_pred,
                                                                                     sudo_golden_out_1=sudo_heads_pred_1,
                                                                                     ori_words=word,
                                                                                     ori_words_length=lengths)
                optim_bia_rl.zero_grad()
                ls_rl_bh.backward()
                optim_bia_rl.step()
                ls_rl_bh = ls_rl_bh.cpu().detach().numpy()
                ls_rl_ep += ls_rl_bh
                # ls1 = ls1.cpu().detach().numpy()
                rewards1 += reward1 * sel.shape[0]
                rewards2 += reward2 * sel.shape[0]
                rewards3 += reward3 * sel.shape[0]
                rewards4 += reward4 * sel.shape[0]
                rewards5 += reward5 * sel.shape[0]
                rewards6 += reward6 * sel.shape[0]

                if kkk % 50 == 0 and kkk != 0:
                    torch.cuda.empty_cache()
                train_num += sel.shape[0]
        if True:
            print('train loss: ', ls_rl_ep)
            print('train reward parser b: ', rewards1 / train_num)
            print('train reward parser c: ', rewards2 / train_num)
            print('train reward parser b^c: ', rewards3 / train_num)
            print('train reward meaning: ', rewards4 / train_num)
            print('train reward fluency: ', rewards5 / train_num)
            print('train UNK score: ', reward6 / train_num)
        for pg in optim_bia_rl.param_groups:
            pg['lr'] *= decay_rate

        if epoch_i >= 0:
            print('Save models to' + args.rl_finetune_seq2seq_save_path)
            torch.save(seq2seq.state_dict(),args.rl_finetune_seq2seq_save_path + '_' + SCORE_PREFIX + str(epoch_i) + '.pt')
            torch.save(network.state_dict(), args.rl_finetune_network_save_path + '_' + SCORE_PREFIX + str(epoch_i) + '.pt')

        ####eval######
        seq2seq.eval()
        network.eval()
        with torch.no_grad():
            ls_rl_ep = rewards1 = rewards2 = rewards3 = rewards4 = rewards5 = 0
            pred_writer_test = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
            if args.treebank == 'ptb':
                pred_filename_test = 'dumped/%spred_test%d' % (SCORE_PREFIX, epoch_i)
                src_filename_test = 'dumped/%ssrc_test%d' % (SCORE_PREFIX, epoch_i)
            elif args.treebank == 'ctb':
                src_filename_test = 'ctb_dumped/%ssrc_test%d' % (SCORE_PREFIX, epoch_i)
                pred_filename_test = 'ctb_dumped/%spred_test%d' % (SCORE_PREFIX, epoch_i)

            pred_writer_test.start(pred_filename_test)

            src_writer_test = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
            src_writer_test.start(src_filename_test)

            nll = 0
            token_num = 0
            kk = 0
            # batch_size_for_eval = 1
            for batch in conllx_data.iterate_batch_tensor(data_test, batch_size):  # batch_size

                kk += 1
                # print('-------' + str(kk) + '-------')
                # if kk > 10:
                #     break
                word, char, pos, heads, types, masks, lengths = batch
                # print(lengths)
                inp = word  # , _ = seq2seq.add_noise(word, lengths)
                sel, pb = seq2seq(inp.long().to(device), LEN=inp.size()[1])
                end_position = torch.eq(sel, END_token).nonzero()
                masks_sel = torch.ones_like(sel, dtype=torch.float)
                lengths_sel = torch.ones_like(lengths).fill_(sel.shape[1])  # sel1.shape[1]-1 TODO: because of end token in the end
                if not len(end_position) == 0:
                    ij_back = -1
                    for ij in end_position:
                        if not (ij[0] == ij_back):
                            lengths_sel[ij[0]] = ij[1]
                            masks_sel[ij[0], ij[1]:] = 0  # -1 TODO: because of end token in the end
                            ij_back = ij[0]
                with torch.no_grad():
                    heads_pred, types_pred = decode(sel, input_char=None, input_pos=None, mask=masks_sel,
                                                    length=lengths_sel,
                                                    leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    if 'stackPtr' == parser_select[0]:
                        sudo_heads_pred, sudo_types_pred = sudo_golden_parser.parsing(sel, None, None, masks_sel,
                                                                                      lengths_sel, beam=1)
                    elif 'bist' == parser_select[0]:
                        str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in one_stc] for
                                   one_stc in sel.cpu().numpy()]
                        stc_pred = list(bist_parser.predict_stcs(str_sel, lengths_sel))
                        sudo_heads_pred = np.array(
                            [[one_w.pred_parent_id for one_w in stc] + [0 for _ in range(sel.shape[1] - len(stc))] for
                             stc in stc_pred])
                    elif parser_select[0] == 'biaffine':
                        sudo_heads_pred, _ = biaffine_parser.decode_mst(sel, input_char=None, input_pos=None, mask=masks_sel,
                                                        length=lengths_sel, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    else:
                        raise ValueError('Error first parser select code!')

                    if 'stackPtr' == parser_select[1]:
                        sudo_heads_pred_1, sudo_types_pred_1 = sudo_golden_parser_1.parsing(sel, None, None, masks_sel,
                                                                                            lengths_sel, beam=1)
                    elif 'bist' == parser_select[1]:
                        str_sel_1 = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in one_stc] for
                                     one_stc in sel.cpu().numpy()]
                        stc_pred_1 = list(bist_parser_1.predict_stcs(str_sel_1, lengths_sel))
                        sudo_heads_pred_1 = np.array(
                            [[one_w.pred_parent_id for one_w in stc] + [0 for _ in range(sel.shape[1] - len(stc))] for
                             stc in stc_pred_1])
                    elif parser_select[1] == 'biaffine':
                        sudo_heads_pred_1, _ = biaffine_parser_1.decode_mst(sel, input_char=None, input_pos=None, mask=masks_sel,
                                                        length=lengths_sel, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    else:
                        raise ValueError('Error second parser select code!')

                ls_rl_bh, reward1, reward2, reward3, reward4, reward5, reward6 = loss_biaf_rl(sel, pb,
                                                                                              predicted_out=heads_pred,
                                                                                              golden_out=heads,
                                                                                              mask_id=END_token,
                                                                                              stc_length_out=lengths_sel,
                                                                                              sudo_golden_out=sudo_heads_pred,
                                                                                              sudo_golden_out_1=sudo_heads_pred_1,
                                                                                              ori_words=word,
                                                                                              ori_words_length=lengths
                                                                                              )  # TODO: (sel, pb, heads)  # heads is replaced by dec_out.long().to(device)

                ls_rl_bh = ls_rl_bh.cpu().detach().numpy()
                ls_rl_ep += ls_rl_bh
                rewards1 += reward1 * sel.shape[0]
                rewards2 += reward2 * sel.shape[0]
                rewards3 += reward3 * sel.shape[0]
                rewards4 += reward4 * sel.shape[0]
                rewards5 += reward5 * sel.shape[0]
                rewards6 += reward6 * sel.shape[0]
                sel = sel.detach().cpu().numpy()
                lengths_sel = lengths_sel.detach().cpu().numpy()
                # print(sel)
                pred_writer_test.write_stc(sel, lengths_sel, symbolic_root=True)
                src_writer_test.write_stc(word, lengths, symbolic_root=True)

                test_num += sel.shape[0]

                for i in range(len(lengths_sel)):
                    nll += sum(pb[i, 1:lengths_sel[i]])
                token_num += sum(lengths_sel) - len(lengths_sel)
            # nll /= token_num

        print('test loss: ', ls_rl_ep)
        print('test reward parser b: ', rewards1 / test_num)
        print('test reward parser c: ', rewards2 / test_num)
        print('test reward parser b^c: ', rewards3 / test_num)
        print('test reward meaning: ', rewards4 / test_num)
        print('test reward fluency: ', rewards5 / test_num)
        print('test UNK score: ', rewards6 / test_num)

        print('test nll: ', nll)

        pred_writer_test.close()
        src_writer_test.close()


if __name__ == '__main__':
    main()
