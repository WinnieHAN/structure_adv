import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import math


class Seq2seq_Model(nn.Module):
    def __init__(self, EMB=8, HID=64, DPr=0.5, vocab_size=None, word_embedd=None, device=None):
        super(Seq2seq_Model, self).__init__()

        self.vocab_size = vocab_size
        self.EMB = EMB
        self.HID = HID
        self.DP = nn.Dropout(DPr)
        self.num_layers = 1
        self.device = device

        self.emb = word_embedd  # nn.Embedding(self.vocab_size + 2, self.EMB)

        self.enc = nn.LSTM(self.EMB, self.HID,  # GRU
                          batch_first=True, bidirectional=True, num_layers=self.num_layers)
        self.dec = nn.LSTM(self.EMB, self.HID * 2,  # because of bidirection
                          batch_first=True, num_layers=self.num_layers)
        self.isLSTM = True
        self.att = nn.Parameter(torch.FloatTensor(self.HID * 2, self.HID * 2))

        self.fc = nn.Linear(self.HID * 2 * 2, self.vocab_size)  # self.vocab_size + 2

        self.init()

    def init(self):
        stdv = 1 / math.sqrt(self.HID * 2)

        self.att.data.uniform_(stdv, -stdv)

    def run_dec(self, dec_inp, out_enc, h):
        dec_inp = self.emb(dec_inp)
        out_dec, h = self.dec(dec_inp, h)
        out_dec = self.DP(out_dec)

        att_wgt = nn.functional.softmax(torch.bmm(torch.matmul(out_dec, self.att), out_enc.transpose(1, 2)), dim=2)
        att_cxt = torch.bmm(att_wgt, out_enc)

        out = torch.cat([out_dec, att_cxt], dim=2)
        out = self.fc(out)

        return out, h

    def forward(self, inp,
                is_tr=False, dec_inp=None, M=4, LEN=None):

        inp = self.emb(inp)
        out_enc, h = self.enc(inp)  # (50, 30, 128) (2, 50, 64)
        out_enc = self.DP(out_enc)
        # h = h.view((1, inp.shape[0], 2 * self.HID * self.num_layers)) #(1, 50, 128)

        if self.isLSTM:
            h = (h[0].view((self.num_layers, 2, inp.shape[0], self.HID)).transpose(1, 2).contiguous().view((self.num_layers, inp.shape[0], 2*self.HID)), h[1].view((self.num_layers, 2, inp.shape[0], self.HID)).transpose(1, 2).contiguous().view((self.num_layers, inp.shape[0], 2*self.HID)) ) # **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`
        else:
            h = h.view((self.num_layers, 2, inp.shape[0], self.HID)).transpose(1, 2).contiguous().view((self.num_layers, inp.shape[0], 2*self.HID))

        # if it is LSTM, output h is (h,c)
        if not dec_inp is None:
            out, _ = self.run_dec(dec_inp, out_enc, h)

            return out

        else:
            # dec_inp = torch.ones((inp.shape[0], 1)).long().cuda() * 2  # id of start: self.vocab_size # START [50,1]
            dec_inp = torch.ones((inp.shape[0], 1)).long().to(self.device) * 2  # id of start: self.vocab_size # START [50,1]
            if is_tr == True:
                if self.isLSTM:
                    h = (torch.cat([h[0] for _ in range(M)], dim=1), torch.cat([h[1] for _ in range(M)], dim=1))  # (1, 200, 128)
                else:
                    h = torch.cat([h for _ in range(M)], dim=1)  # (1, 200, 128)
                out_enc = torch.cat([out_enc for _ in range(M)], dim=0)  # (200, 30,128)

                dec_inp = torch.cat([dec_inp for _ in range(M)], dim=0)  # (200, 1)

            outs, sels, pbs = [], [], []
            for ii in range(LEN):
                out, h = self.run_dec(dec_inp, out_enc, h)  #(200, 1) (200,30,128) (1, 200, 128)  ||(200,1) (200, 15, 128) (1, 200, 128)
                outs.append(out)

                if is_tr == True:
                    temp = nn.functional.softmax(out[:, 0, :], dim=1)
                    dec_inp = torch.multinomial(temp, 1)  # (200, 1) TODO: hanwj. why here is [:,0,:].  every time we use the same tensor to softmax?
                else:
                    dec_inp = torch.argmax(out, dim=2, keepdim=False)
                sels.append(dec_inp)  # dec_inp: [200, 1]

                pb = torch.gather(nn.functional.log_softmax(out, dim=2), 2, dec_inp.view((dec_inp.shape[0], 1, 1)))  # TODO: dec_inp ??
                pb = pb.view((pb.shape[0], 1))  # pb: [200, 1]
                pbs.append(pb)

            out = torch.cat(outs, dim=1)  # (200, 15 ,12)
            sel = torch.cat(sels, dim=1)   # (200, 15)  ids
            pb = torch.cat(pbs, dim=1)

            if is_tr == True:
                return out, sel, pb
            else:
                return sel, pb

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        """
        # words, lengths = self.word_shuffle(words, lengths)
        # words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        word_blank = 0
        blank_index = 0  # should be defined TODO:
        pad_index = 0  # should be defined TODO:
        # define words to blank
        keep = np.random.rand(x.size(0), x.size(1)) >= word_blank
        # do not blank the start sentence symbol TODO:
        sentences = []
        for i in range(l.size(0)):
            words = x[i, :l[i]].tolist()
            # randomly blank words from the input
            new_s = [w if keep[i, j] else blank_index for j, w in enumerate(words)]
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(x.size(0), x.size(1)).fill_(pad_index)
        for i in range(l.size(0)):
            x2[i, :l[i]].copy_(torch.LongTensor(sentences[i]))

        return x2, l

    # new method of adding noising  # TODO: hanwj
    def word_shuffle(self, x, l, lang_id):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
            scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l, lang_id):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.params.eos_index)
            assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)   # hanwj
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank1(self, x, l, lang_id):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        # be sure to blank entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[word_idx[j, i], i] else self.params.blank_index for j, w in enumerate(words)]
            new_s.append(self.params.eos_index)
            assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_stop_token(self, masks, lengths):
        wget = masks.data.detach()
        wget[range(lengths.shape[0]), lengths] = 1.0
        return wget