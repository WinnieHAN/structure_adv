from nltk.translate.bleu_score import sentence_bleu as BLEU
import numpy as np
import torch.nn as nn
import torch
# ref = [[1, 2, 3, 4, 5, 6]]
# cnd = [1, 3, 4, 5, 6]
# bleu = BLEU(ref, cnd)
#
# print('BLEU: %.4f%%' % (bleu * 100))


def get_bleu(out, dec_out, vocab_size):
    out = out.tolist()
    dec_out = dec_out.tolist()

    if vocab_size + 1 in out:
        cnd = out[:out.index(vocab_size + 1)]
    else:
        cnd = out

    if vocab_size + 1 in dec_out:
        ref = [dec_out[:dec_out.index(vocab_size + 1)]]
    else:
        ref = [dec_out]

    bleu = BLEU(ref, cnd)

    return bleu



class LossRL(nn.Module):
    def __init__(self):
        super(LossRL, self).__init__()

        self.bl = 0
        self.bn = 0

    def forward(self, sel, pb, dec_out, stc_length, vocab_size):
        ls = 0
        cnt = 0

        sel = sel.detach().cpu().numpy()
        dec_out = dec_out.cpu().numpy()

        batch = sel.shape[0]
        bleus = []
        for i in range(batch):
            bleu = get_bleu(sel[i], dec_out[i], vocab_size)

            bleus.append(bleu)
        bleus = np.asarray(bleus)

        wgt = np.asarray([1 for _ in range(batch)])
        for j in range(stc_length):
            ls += (- pb[:, j] *
                   torch.from_numpy(bleus - self.bl).float().cuda() *
                   torch.from_numpy(wgt.astype(float)).float().cuda()).sum()
            cnt += np.sum(wgt)

            wgt = wgt.__and__(sel[:, j] != vocab_size + 1)

        ls /= cnt

        bleu = np.average(bleus)
        self.bl = (self.bl * self.bn + bleu) / (self.bn + 1)
        self.bn += 1

        return ls


def get_reward(out, dec_out):
    # out = out.tolist()
    # dec_out = dec_out.tolist()

    dda = sum([1 if out[i]==dec_out[i] else 0 for i in range(min(len(out), len(dec_out)))])

    return dda


class LossBiafRL(nn.Module):
    def __init__(self):
        super(LossBiafRL, self).__init__()

        self.bl = 0
        self.bn = 0

    def forward(self, sel, pb, predicted_out, golden_out, mask_id):
        ls = 0
        cnt = 0

        stc_length = sel.shape[1]
        sel = sel.detach().cpu().numpy()
        # predicted_out = predicted_out.cpu().numpy()
        golden_out = golden_out.cpu().numpy()

        batch = sel.shape[0]
        bleus = []
        for i in range(batch):  #batch
            bleu = get_reward(predicted_out[i], golden_out[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.

            bleus.append(bleu)
        bleus = np.asarray(bleus)

        wgt = np.asarray([1 for _ in range(batch)])
        for j in range(stc_length):
            ls += (- pb[:, j] *
                   torch.from_numpy(bleus - self.bl).float().cuda() *
                   torch.from_numpy(wgt.astype(float)).float().cuda()).sum()
            cnt += np.sum(wgt)

            wgt = wgt.__and__(sel[:, j] != mask_id)  # vocab_size + 1

        ls /= cnt

        bleu = np.average(bleus)
        self.bl = (self.bl * self.bn + bleu) / (self.bn + 1)
        self.bn += 1

        return ls