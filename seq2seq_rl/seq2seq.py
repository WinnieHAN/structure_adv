import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch as T
import math

class Seq2seq_Model(nn.Module):
    def __init__(self, EMB=8, HID=64, DPr=0.5, vocab_size=None, word_embedd=None):
        super(Seq2seq_Model, self).__init__()

        self.vocab_size = vocab_size
        self.EMB = EMB
        self.HID = HID
        self.DP = nn.Dropout(DPr)

        self.emb = word_embedd  # nn.Embedding(self.vocab_size + 2, self.EMB)

        self.enc = nn.GRU(self.EMB, self.HID,
                          batch_first=True, bidirectional=True)
        self.dec = nn.GRU(self.EMB, self.HID * 2,
                          batch_first=True)

        self.att = nn.Parameter(T.FloatTensor(self.HID * 2, self.HID * 2))

        self.fc = nn.Linear(self.HID * 2 * 2, self.vocab_size + 2)

        self.init()

    def init(self):
        stdv = 1 / math.sqrt(self.HID * 2)

        self.att.data.uniform_(stdv, -stdv)

    def run_dec(self, dec_inp, out_enc, h):
        dec_inp = self.emb(dec_inp)
        out_dec, h = self.dec(dec_inp, h)
        out_dec = self.DP(out_dec)

        att_wgt = nn.functional.softmax(T.bmm(T.matmul(out_dec, self.att), out_enc.transpose(1, 2)), dim=2)
        att_cxt = T.bmm(att_wgt, out_enc)

        out = T.cat([out_dec, att_cxt], dim=2)
        out = self.fc(out)

        return out, h

    def forward(self, inp,
                is_tr=False, dec_inp=None, M=4, LEN=None):

        inp = self.emb(inp)
        out_enc, h = self.enc(inp)  # (50, 30, 128) (2, 50, 64)
        out_enc = self.DP(out_enc)
        h = h.view((1, inp.shape[0], 2 * self.HID))  #(1, 50, 128)

        if not dec_inp is None:
            out, _ = self.run_dec(dec_inp, out_enc, h)

            return out

        else:
            dec_inp = T.ones((inp.shape[0], 1)).long().cuda() * 2  # id of start: self.vocab_size # START [50,1]

            if is_tr == True:
                h = T.cat([h for _ in range(M)], dim=1)  # (1, 200, 128)
                out_enc = T.cat([out_enc for _ in range(M)], dim=0)  # (200, 30,128)

                dec_inp = T.cat([dec_inp for _ in range(M)], dim=0)  # (200, 1)

            outs, sels, pbs = [], [], []
            for ii in range(LEN):
                out, h = self.run_dec(dec_inp, out_enc, h)  #(200, 1) (200,30,128) (1, 200, 128)  ||(200,1) (200, 15, 128) (1, 200, 128)
                outs.append(out)

                if is_tr == True:
                    dec_inp = T.multinomial(nn.functional.softmax(out[:, 0, :], dim=1), 1)  # (200, 1) TODO: hanwj. why here is [:,0,:].  every time we use the same tensor to softmax?
                else:
                    dec_inp = T.argmax(out, dim=2, keepdim=False)
                sels.append(dec_inp)  # dec_inp: [200, 1]

                pb = T.gather(nn.functional.log_softmax(out, dim=2), 2, dec_inp.view((dec_inp.shape[0], 1, 1)))  # TODO: dec_inp ??
                pb = pb.view((pb.shape[0], 1))  # pb: [200, 1]
                pbs.append(pb)

            out = T.cat(outs, dim=1)  # (200, 15 ,12)
            sel = T.cat(sels, dim=1)   # (200, 15)  ids
            pb = T.cat(pbs, dim=1)

            if is_tr == True:
                return out, sel, pb
            else:
                return sel

    def add_noise(self, words):
        # TODO
        return words