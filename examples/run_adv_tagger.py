from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for POS tagging.
"""

import sys, os

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
from seq2seq_rl.rl import LossRL, TagLossBiafRL, get_bleu, get_correct
# import pickle
from nltk.tag.senna import SennaTagger
from nltk.tag import StanfordPOSTagger
import global_variables

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

    parser.add_argument('--direct_eval', action='store_true', help='direct eval without generation process')
    parser.add_argument('--port', type=int, default=10048, help='localhost port for berscore server')
    parser.add_argument('--z1_weight', type=float, default=1.0, help='reward weight of z1')
    parser.add_argument('--z2_weight', type=float, default=1.0, help='reward weight of z2')
    parser.add_argument('--z3_weight', type=float, default=0.01, help='reward weight of z3')
    parser.add_argument('--mp_weight', type=float, default=100, help='reward weight of meaning preservation')
    parser.add_argument('--ppl_weight', type=float, default=0.001, help='reward weight of ppl')
    parser.add_argument('--unk_weight', type=float, default=100, help='reward weight of unk rate')
    parser.add_argument('--prefix', type=str, default='/home/zhanglw/code/structure_adv/test_')

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

    # rl weight
    global_variables.Z1_REWARD_WEIGHT = args.z1_weight
    global_variables.Z2_REWARD_WEIGHT = args.z2_weight
    global_variables.Z3_REWARD_WEIGHT = args.z3_weight
    global_variables.MP_REWARD_WEIGHT = args.mp_weight
    global_variables.PPL_REWARD_WEIGHT = args.ppl_weight
    global_variables.UNK_REWARD_WEIGHT = args.unk_weight

    global_variables.PREFIX = args.prefix
    SCORE_PREFIX = args.prefix.split('/')[-1]

    port = args.port

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


    print('Train seq2seq model using rl with reward of biaffine.')

    seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(2) + '.pt'))
    seq2seq.to(device)
    network.load_state_dict(torch.load(args.network_load_path + str(2) + '.pt'))
    network.to(device)

    sudo_golden_tagger = SennaTagger(parser_b)
    sudo_golden_tagger_1 = StanfordPOSTagger(model_filename=parser_c + '/models/english-bidirectional-distsim.tagger',
                                             path_to_jar=parser_c + '/stanford-postagger.jar')

    EPOCHS = num_epochs
    DECAY = 0.97
    M = 1  # this is the size of beam searching ?
    seq2seq.emb.weight.requires_grad = False
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_bia_rl = torch.optim.Adam(parameters_need_update, lr=learning_rate, )  #1e-5 0.00005
    loss_biaf_rl = TagLossBiafRL(device=device, word_alphabet=word_alphabet, vocab_size=num_words, port=port).to(device)

    num_batches = num_data / batch_size + 1

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


    for epoch_i in range(EPOCHS):
        print('======='+str(epoch_i)+'=========')
        ls_rl_ep = rewards1 = rewards2 = rewards3 = rewards4 = rewards5 = rewards6 = cnt = 0.0
        network.eval()  # only train seq2seq
        seq2seq.train()
        seq2seq.emb.weight.requires_grad = False
        END_token = word_alphabet.instance2index['_PAD']
        print('END_token: '+str(END_token))
        import time
        start_time = time.time()
        print('num_batches: ', str(num_batches))
        for kkk in range(1, num_batches + 1): #num_batches
            print('-train--'+str(kkk)+'---')
            word, char, labels, _, _, masks, lengths = conllx_data.get_batch_tensor(data_train, batch_size, unk_replace=unk_replace)
            inp = word
            _, sel, pb = seq2seq(inp.long().to(device), is_tr=True, M=M, LEN=inp.size()[1])
            sel1 = sel.data.detach()
            try:
                end_position = torch.eq(sel1, END_token).nonzero()
            except RuntimeError:
                continue
                print(sel1)
            masks_sel = torch.ones_like(sel1, dtype=torch.float)
            lengths_sel = torch.ones_like(lengths).fill_(sel1.shape[1])
            if not len(end_position)==0:
                ij_back = -1
                for ij in end_position:
                    if not (ij[0]==ij_back):
                        lengths_sel[ij[0]] = ij[1]
                        masks_sel[ij[0], ij[1]:] = 0
                        ij_back = ij[0]

            with torch.no_grad():
                try:
                    char1 = word_to_chars_tensor(char.shape, sel1, lengths_sel, word_alphabet, char_alphabet)
                    tags_pred, corr = network.decode(sel1, char1, target=labels, mask=masks_sel,
                                                 leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
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
                except:
                    print('IndexError Maybe sel1: ', sel1)
                    print('IndexError Maybe: ', sel1.data.cpu().numpy())
                    print(masks_sel)
                    continue
            ls_rl_bh, reward1, reward2, reward3, reward4, reward5, reward6 = loss_biaf_rl(sel, pb, predicted_out=tags_pred, golden_out=labels, mask_id=END_token,
                                                                          stc_length_out=lengths_sel, sudo_golden_out=sudo_tags_pred, sudo_golden_out_1=sudo_tags_pred_1,
                                                                          ori_words=word, ori_words_length=lengths)
            optim_bia_rl.zero_grad()
            ls_rl_bh.backward()
            optim_bia_rl.step()
            ls_rl_bh = ls_rl_bh.cpu().detach().numpy()
            ls_rl_ep += ls_rl_bh
            rewards1 += reward1 * word.size()[0]
            rewards2 += reward2 * word.size()[0]
            rewards3 += reward3 * word.size()[0]
            rewards4 += reward4 * word.size()[0]
            rewards5 += reward5 * word.size()[0]
            rewards6 += reward6 * word.size()[0]
            cnt += word.size()[0]

        print('train loss: ', ls_rl_ep)
        print('train reward parser b: ', rewards1 / cnt)
        print('train reward parser c: ', rewards2 / cnt)
        print('train reward parser b^c: ', rewards3 / cnt)
        print('train reward meaning: ', rewards4 / cnt)
        print('train reward fluency: ', rewards5 / cnt)
        print('train UNK rate: ', rewards6 / cnt)
        for pg in optim_bia_rl.param_groups:
            pg['lr'] *= DECAY

        if epoch_i >= 0:
            torch.save(seq2seq.state_dict(), args.rl_finetune_seq2seq_save_path + '_' + SCORE_PREFIX + str(epoch_i) + '.pt')
            torch.save(network.state_dict(), args.rl_finetune_network_save_path + '_' + SCORE_PREFIX + str(epoch_i) + '.pt')

        ####eval######
        if True:
            seq2seq.eval()
            network.eval()
            ls_rl_ep = rewards1 = rewards2 = rewards3 = rewards4 = rewards5 = \
                zlw_rewards1 = zlw_rewards2 = zlw_rewards3 = rewardsall1 = rewardsall2 = rewardsall3 = 0
            pred_writer_test = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
            pred_filename_test = 'tagging_dumped/' + '_' + SCORE_PREFIX +'pred_test%d' % (epoch_i)
            src_filename_test = 'tagging_dumped/' + '_' + SCORE_PREFIX +'src_test%d' % (epoch_i)

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
            batch_size_for_eval = 3
            for batch in conllx_data.iterate_batch_tensor(data_test, batch_size_for_eval):  # batch_size
                kk += 1
                # print('-------'+str(kk)+'-------')
                if kk > 2:
                    break
                # if kk==7:
                #     print('error here')
                word, char, labels, _, _, masks, lengths = batch
                with torch.no_grad():
                    if not args.direct_eval:
                        inp = word  #, _ = seq2seq.add_noise(word, lengths)
                        sel, pb = seq2seq(inp.long().to(device), LEN=inp.size()[1])
                        end_position = torch.eq(sel, END_token).nonzero()
                        masks_sel = torch.ones_like(sel, dtype=torch.float)
                        lengths_sel = torch.ones_like(lengths).fill_(sel.shape[1])
                        if not len(end_position) == 0:
                            ij_back = -1
                            for ij in end_position:
                                if not (ij[0]==ij_back):
                                    lengths_sel[ij[0]] = ij[1]
                                    masks_sel[ij[0], ij[1]:] = 0
                                    ij_back = ij[0]
                    else:
                        sel = word
                        pb = torch.ones_like(sel, dtype=torch.float).fill_(0)
                        lengths_sel = lengths
                        masks_sel = masks
                        char1 = char

                        tags_pred, corr = network.decode(sel, char1, target=labels, mask=masks_sel,
                                                         leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    sel_num = sel.cpu().numpy()
                    sentence_max_length = len(sel_num[0])
                    str_sel = [[word_alphabet.get_instance(one_word).encode('utf-8') for one_word in sel_num[one_stc_id][:lengths_sel[one_stc_id]]] for
                               one_stc_id in range(len(sel_num))]
                    if 'tagger0' in parser_select:
                        temp = [[pos_alphabet.get_index(j[1]) for j in sudo_golden_tagger.tag(i)] for i in str_sel]
                        sudo_tags_pred = [i+[1 for _ in range(sentence_max_length-len(i))] for i in temp]
                    if 'tagger1' in parser_select:
                        temp = [[pos_alphabet.get_index(j[1]) for j in sudo_golden_tagger_1.tag(i)] for i in str_sel]
                        sudo_tags_pred_1 = [i+[1 for _ in range(sentence_max_length-len(i))] for i in temp]

                    res = loss_biaf_rl.forward_verbose(sel, pb, predicted_out=tags_pred,golden_out=labels,
                                                        mask_id=END_token,stc_length_out=lengths_sel,
                                                        sudo_golden_out=sudo_tags_pred,
                                                        sudo_golden_out_1=sudo_tags_pred_1,
                                                        ori_words=word,
                                                        ori_words_length=lengths)

                ls_rl_bh = res['loss'].item()
                ls_rl_ep += ls_rl_bh * word.size(0)
                rewards1 += res['sum_me1']
                rewards2 += res['sum_me2']
                rewards3 += res['sum_me3']
                rewards4 += res['sum_me4']
                rewards5 += res['sum_me5']
                rewardsall1 += res['sum_me1all']
                rewardsall2 += res['sum_me2all']
                rewardsall3 += res['sum_me3all']
                zlw_rewards1 += res['cnt_me1']
                zlw_rewards2 += res['cnt_me2']
                zlw_rewards3 += res['cnt_me3']

                sel = sel.detach().cpu().numpy()
                lengths_sel = lengths_sel.detach().cpu().numpy()
                # print(sel)
                pred_writer_test.write_stc(sel, lengths_sel, symbolic_root=False)
                src_writer_test.write_stc(word, lengths, symbolic_root=False)
                pred_parse_writer_testA.write(sel, sel, tags_pred, tags_pred, lengths_sel, symbolic_root=False)  # word, pos, head, type, lengths,
                pred_parse_writer_testB.write(sel, sel, sudo_tags_pred, sudo_tags_pred, lengths_sel, symbolic_root=False)  # word, pos, head, type, lengths,
                pred_parse_writer_testC.write(sel, sel, sudo_tags_pred_1, sudo_tags_pred_1, lengths_sel, symbolic_root=False)  # word, pos, head, type, lengths,

                for i in range(len(lengths_sel)):
                    nll += sum(pb[i, 1:lengths_sel[i]])
                token_num += sum(lengths_sel)#-len(lengths_sel)

            rewards1 = rewards1 * 1.0 / sum(data_test[1])
            rewards2 = rewards2 * 1.0 / sum(data_test[1])
            rewards3 = rewards3 * 1.0 / sum(data_test[1])
            rewards4 = rewards4 * 1.0 / sum(data_test[1])
            rewards5 = rewards5 * 1.0 / sum(data_test[1])
            rewardsall1 = rewardsall1 * 1.0 / sum(data_test[1])
            rewardsall2 = rewardsall2 * 1.0 / sum(data_test[1])
            rewardsall3 = rewardsall3 * 1.0 / sum(data_test[1])
            zlw_rewards1 = zlw_rewards1 * 1.0 / token_num
            zlw_rewards2 = zlw_rewards2 * 1.0 / token_num
            zlw_rewards3 = 1 -zlw_rewards3 * 1.0 / token_num
            print('test loss: ', ls_rl_ep)
            print('test reward parser b: ', rewards1)
            print('test zlw reward parser b: ', zlw_rewards1)
            print('test reward parser c: ', rewards2)
            print('test zlw reward parser c: ', zlw_rewards2)
            print('test reward parser b^c: ', rewards3)
            print('test zlw reward parser b^c: ', zlw_rewards3)
            print('test reward meaning: ', rewards4)
            print('test reward fluency: ', rewards5)

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
