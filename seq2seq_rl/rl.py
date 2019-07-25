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


def get_correct(out, dec_out, length):
    stc_dda = sum([1 if out[i] == dec_out[i] else 0 for i in range(1, length)])
    return stc_dda, length-1

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


def get_reward(out, dec_out, length_out):
    # out = out.tolist()
    # dec_out = dec_out.tolist()

    stc_dda = sum([1 if out[i]==dec_out[i] else 0 for i in range(1, length_out)])

    return stc_dda


class LossBiafRL(nn.Module):
    def __init__(self):
        super(LossBiafRL, self).__init__()

        self.bl = 0
        self.bn = 0

    def forward(self, sel, pb, predicted_out, golden_out, mask_id, stc_length_out, sudo_golden_out, sudo_golden_out_1):
        ls = 0
        cnt = 0

        stc_length = sel.shape[1]
        sel = sel.detach().cpu().numpy()
        golden_out = golden_out.cpu().numpy()

        batch = sel.shape[0]
        bleus = []
        for i in range(batch):  #batch
            bleu = get_reward(predicted_out[i], golden_out[i], stc_length_out[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
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

        ########
        ls1 = 0
        cnt1 = 0

        batch = sel.shape[0]
        bleus1 = []
        for i in range(batch):  #batch
            bleu = get_reward(predicted_out[i], sudo_golden_out[i], stc_length_out[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            bleus1.append(bleu)
        bleus1 = np.asarray(bleus1)

        wgt1 = np.asarray([1 for _ in range(batch)])
        for j in range(stc_length):
            ls1 += (- pb[:, j] *
                   torch.from_numpy(bleus1).float().cuda() *
                   torch.from_numpy(wgt1.astype(float)).float().cuda()).sum()
            cnt1 += np.sum(wgt1)

            wgt1 = wgt1.__and__(sel[:, j] != mask_id)  # vocab_size + 1

        ls1 /= cnt1
        bleu1 = np.average(bleus1)

        ########
        ls2 = 0
        cnt2 = 0

        batch = sel.shape[0]
        bleus2 = []
        for i in range(batch):  #batch
            bleu = get_reward(predicted_out[i], sudo_golden_out_1[i], stc_length_out[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            bleus2.append(bleu)
        bleus2 = np.asarray(bleus2)

        wgt2 = np.asarray([1 for _ in range(batch)])
        for j in range(stc_length):
            ls1 += (- pb[:, j] *
                   torch.from_numpy(bleus2).float().cuda() *
                   torch.from_numpy(wgt2.astype(float)).float().cuda()).sum()
            cnt2 += np.sum(wgt2)

            wgt2 = wgt2.__and__(sel[:, j] != mask_id)  # vocab_size + 1

        ls2 /= cnt2
        bleu2 = np.average(bleus2)

        loss = -ls1 -ls2  #- ls + ls1
        return loss, ls, ls1, bleu, bleu1