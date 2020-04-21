__author__= 'Ehaschia'
"""
This file is used to generate advasial sample from trained seq2seq model
"""

import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.append(".")
sys.path.append("..")

import argparse
import uuid

import numpy as np
import torch
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvBiAffine
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from seq2seq_rl.seq2seq import Seq2seq_Model
from stack_parser_eval import third_party_parser
import pickle
from bist_parser.barchybrid.src.arc_hybrid import ArcHybridLSTM
import global_variables

uid = uuid.uuid4().hex[:6]


def parse_diff(out, dec_out, length_out):
    stc_dda = sum([0 if out[i] == dec_out[i] else 1 for i in range(1, length_out)])
    return stc_dda / (length_out - 1)


def unk_rate(sent):
    return sent.count(0) / len(sent)


def main():
    args_parser = argparse.ArgumentParser(description='Generate sentence by seq2seq model')
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
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
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
    args_parser.add_argument('--prefix', type=str, default='')
    args = args_parser.parse_args()

    logger = get_logger("Sentence Generator")

    parser_select = ['stackPtr', 'bist']

    mode = args.mode
    obj = args.objective
    decoding = args.decode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    model_path = args.model_path
    model_name = args.model_name
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    num_layers = args.num_layers
    num_filters = args.num_filters

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

    global_variables.PREFIX = args.prefix


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
    END_token = word_alphabet.instance2index['_PAD']


    logger.info("Reading Data")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    data_train = conllx_data.read_data_to_tensor(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                 symbolic_root=True, device=device)
    num_data = sum(data_train[1])

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32)
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
                                          window, mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                          embedd_word=word_table, embedd_char=char_table,
                                          p_in=0.0, p_out=0.0, p_rnn=[0.0, 0.0], biaffine=True, pos=use_pos,
                                          char=use_char)
    else:
        raise RuntimeError('Unknown objective: %s' % obj)

    network = network.to(device)


    # Build generate model
    shared_word_embedd = network.return_word_embedd()
    shared_word_embedd.weight.requires_grad = False
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words,
                            word_embedd=shared_word_embedd, device=device).to(device)
    seq2seq.emb.weight.requires_grad = False
    print(seq2seq)

    # load pretrained model
    seq2seq.load_state_dict(torch.load(args.rl_finetune_seq2seq_load_path + args.prefix + '.pt'))
    seq2seq.to(device)
    network.load_state_dict(torch.load(args.rl_finetune_network_load_path + args.prefix + '.pt'))
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
                                                  embedd_word=word_table, embedd_char=char_table,p_in=0.0, p_out=0.0,
                                                  p_rnn=[0.0, 0.0], biaffine=True, pos=use_pos, char=use_char)

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
                                                  embedd_word=word_table, embedd_char=char_table,p_in=0.0, p_out=0.0,
                                                  p_rnn=[0.0, 0.0], biaffine=True, pos=use_pos, char=use_char)

        biaffine_parser_1 = biaffine_parser_1.to(device)
        biaffine_parser_1.load_state_dict(torch.load('models/parsing/biaffine2/network.pt'))
        biaffine_parser_1.eval()


    # Begin generate

    network.eval()
    seq2seq.eval()
    generation_res = []
    
    with torch.no_grad():
        for batch in conllx_data.iterate_batch_tensor(data_train, batch_size):  # batch_size
            word, char, pos, heads, types, masks, lengths = batch

            inp = word
            sel, pb = seq2seq(inp.long().to(device), LEN=inp.size()[1])
            end_position = torch.eq(sel, END_token).nonzero()
            masks_sel = torch.ones_like(sel, dtype=torch.float)
            lengths_sel = torch.ones_like(lengths).fill_(
                sel.shape[1])  # sel1.shape[1]-1 TODO: because of end token in the end
            if not len(end_position) == 0:
                ij_back = -1
                for ij in end_position:
                    if not (ij[0] == ij_back):
                        lengths_sel[ij[0]] = ij[1]
                        masks_sel[ij[0], ij[1]:] = 0  # -1 TODO: because of end token in the end
                        ij_back = ij[0]

            # current parsing result
            heads_pred, types_pred = network.decode_mst(sel, input_char=None, input_pos=None, mask=masks_sel,
                                                        length=lengths_sel, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
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
                                                                length=lengths_sel,
                                                                leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
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
                sudo_heads_pred_1, _ = biaffine_parser_1.decode_mst(sel, input_char=None, input_pos=None,
                                                                    mask=masks_sel,
                                                                    length=lengths_sel,
                                                                    leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            else:
                raise ValueError('Error second parser select code!')

            # np_sel = sel.cpu().numpy()
            for i in range(batch[0].size()[0]):
                # print("Different between parser A and parser B: \t"
                #       + str(parse_diff(heads_pred[i], sudo_heads_pred[i], lengths_sel[i]))
                #       + " Different between parser A and parser C: \t"
                #       + str(parse_diff(heads_pred[i], sudo_heads_pred_1[i], lengths_sel[i]))
                #       + " Different between parser B and parser C: \t"
                #       + str(parse_diff(sudo_heads_pred[i], sudo_heads_pred_1[i], lengths_sel[i])))


                # # get current length
                # gen_length = np.where(np_sel == 1)
                if (parse_diff(heads_pred[i], sudo_heads_pred[i], lengths_sel[i]) != 0
                        and parse_diff(heads_pred[i], sudo_heads_pred_1[i], lengths_sel[i]) != 0
                        and parse_diff(sudo_heads_pred[i], sudo_heads_pred_1[i], lengths_sel[i]) == 0):
                    generation_res.append((word[i].cpu().numpy().tolist(),
                                           sel[i].cpu().numpy().tolist(),
                                           heads_pred[i].tolist(),
                                           sudo_heads_pred[i].tolist(),
                                           lengths_sel[i].item()))
                else:
                    pass
            # if len(generation_res) > 10:
            #     break
    # save generation result back to
    save_path = 'data/ptb/gen' + args.prefix + '.conllu'
    writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    writer.start(save_path)
    raw_sents = []
    gen_sents = []
    pred_trees = []
    golden_trees = []
    gen_lengths = []
    max_len = 0
    for res in generation_res:
        raw_sent, gen_sent, pred_tree, golden_tree, gen_length = res
        raw_sents.append(raw_sent)
        gen_sents.append(gen_sent)
        pred_trees.append(pred_tree)
        golden_trees.append(golden_tree)
        gen_lengths.append(gen_length)
        if gen_length > max_len:
            max_len = gen_length

    type = [[3]*max_len for _ in range(len(gen_sents))]
    pos = [[4]*max_len for _ in range(len(gen_sents))]
    writer.write(np.array(gen_sents),
                 np.array(pos),
                 np.array(golden_trees),
                 np.array(type),
                 np.array(gen_lengths), symbolic_root=True)


if __name__ == '__main__':
    main()
