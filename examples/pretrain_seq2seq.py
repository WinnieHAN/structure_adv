from __future__ import print_function

import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')   # Try setting the system default encoding as utf-8 at the start of the script, so that all strings are encoded using that. Or there will be UnicodeDecodeError: 'ascii' codec can't decode byte...

sys.path.append(".")
sys.path.append("..")

import argparse
import uuid

import numpy as np
import torch
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvBiAffine
from neuronlp2 import utils
from seq2seq_rl.seq2seq import Seq2seq_Model
from seq2seq_rl.rl import get_bleu, get_correct


uid = uuid.uuid4().hex[:6]

def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU', 'FastLSTM'], help='architecture of rnn',
                             required=True)
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
    args_parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0.,
                             help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'],
                             help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters',
                             required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')
    args_parser.add_argument('--dev')
    args_parser.add_argument('--test')
    args_parser.add_argument('--model_path', help='path for pretrained biaffine model file.', required=True)

    args_parser.add_argument('--seq2seq_save_path', default='models/seq2seq/seq2seq_save_model', type=str,
                             help='seq2seq_save_path')
    args_parser.add_argument('--network_save_path', default='models/seq2seq/network_save_model', type=str,
                             help='network_save_path')

    args = args_parser.parse_args()

    logger = get_logger("pretrain_seq2seq")

    # SEED = 0
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    model_path = args.model_path
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    num_layers = args.num_layers
    num_filters = args.num_filters
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
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                             data_paths=[dev_path,
                                                                                                         test_path],
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
    network = BiRecurrentConvBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters,
                                      window,
                                      mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                      embedd_word=word_table, embedd_char=char_table,
                                      p_in=p_in, p_out=p_out, p_rnn=p_rnn, biaffine=True, pos=use_pos,
                                      char=use_char)
    network.load_state_dict(torch.load('models/parsing/biaffine/network.pt'))
    network = network.to(device)

    num_batches = num_data / batch_size + 1

    # Pretrain seq2seq model using denoising autoencoder. model name: seq2seq model
    print('Pretrain seq2seq model using denoising autoencoder.')
    EPOCHS = args.num_epochs  # 150
    DECAY = 0.97
    shared_word_embedd = network.return_word_embedd()
    shared_word_embedd.weight.requires_grad = False
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words,
                            word_embedd=shared_word_embedd, device=device).to(device)  # debug hanwj
    seq2seq.emb.weight.requires_grad = False
    print(seq2seq)

    loss_seq2seq = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_seq2seq = torch.optim.Adam(parameters_need_update, lr=args.learning_rate)
    # seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(2) + '.pt'))
    # seq2seq.to(device)
    # network.load_state_dict(torch.load(args.network_load_path + str(2) + '.pt'))
    # network.to(device)


    for i in range(EPOCHS):
        ls_seq2seq_ep = 0
        seq2seq.train()
        network.train()
        seq2seq.emb.weight.requires_grad = False
        print('----------' + str(i) + ' iter----------')
        for _ in range(1, num_batches + 1):
            word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_train, batch_size,
                                                                                         unk_replace=unk_replace)
            inp, _ = seq2seq.add_noise(word, lengths)
            dec_out = word
            dec_inp = torch.cat((word[:, 0:1], word[:, 0:-1]), dim=1)
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
        seq2seq.eval()
        network.eval()
        bleu_ep = 0
        acc_numerator_ep = 0
        acc_denominator_ep = 0
        testi = 0
        for batch in conllx_data.iterate_batch_tensor(data_dev, batch_size):
            word, char, pos, heads, types, masks, lengths = batch
            inp = word
            # inp, _ = seq2seq.add_noise(word, lengths)
            dec_out = word
            with torch.no_grad():
                sel, _ = seq2seq(inp.long().to(device), LEN=inp.size()[1])
                sel = sel.detach().cpu().numpy()
            dec_out = dec_out.cpu().numpy()

            bleus = []
            for j in range(sel.shape[0]):
                bleu = get_bleu(sel[j], dec_out[j], num_words)  # sel
                bleus.append(bleu)
                numerator, denominator = get_correct(sel[j], dec_out[j], num_words)
                acc_numerator_ep += numerator
                acc_denominator_ep += denominator  # .detach().cpu().numpy()
            bleu_bh = np.average(bleus)
            bleu_ep += bleu_bh
            testi += 1
        bleu_ep /= testi  # num_batches
        print('testi: ', testi)
        print('Valid bleu: %.4f%%' % (bleu_ep * 100))
        # print(acc_denominator_ep)
        print('Valid acc: %.4f%%' % ((acc_numerator_ep * 1.0 / acc_denominator_ep) * 100))
        torch.save(seq2seq.state_dict(), args.seq2seq_save_path + str(i) + '.pt')
        torch.save(network.state_dict(), args.network_save_path + str(i) + '.pt')
    # Pretrain seq2seq model using token wise adv examples. model name: seq2seq model
    # print('Pretrain seq2seq model using token wise adv examples.')

if __name__ == '__main__':
    main()