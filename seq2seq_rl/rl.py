from nltk.translate.bleu_score import sentence_bleu as BLEU
import numpy as np
import torch.nn as nn
import torch, os, codecs
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


class LossBiafRL(nn.Module):
    def __init__(self, device, word_alphabet):
        super(LossBiafRL, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet

    def get_reward(self, out, dec_out, length_out, ori_words, ori_words_length):
        # out = out.tolist()
        # dec_out = dec_out.tolist()
        # word_alphabet.get_instance(one_word).encode('utf-8')
        reward = 0
        stc_dda = sum([0 if out[i] == dec_out[i] else 1 for i in range(1, length_out)])

        reward = stc_dda

        return reward

    def write_text(self, ori_words, ori_words_length, sel, stc_length_out):
        condsf = 'cands.txt'
        refs = 'refs.txt'
        oris = [[self.word_alphabet.get_instance(ori_words[si, wi]).encode('utf-8') for wi in range(1, ori_words_length[si])] for si in range(len(ori_words))]
        preds = [[self.word_alphabet.get_instance(sel[si, wi]).encode('utf-8') for wi in range(1, stc_length_out[si])] for si in range(len(sel))]

        wf = codecs.open(condsf, 'w', encoding='utf8')
        preds_tmp = [' '.join(i) for i in preds]
        preds_s = '\n'.join(preds_tmp)
        wf.write(preds_s)
        wf.close()

        wf = codecs.open(refs, 'w', encoding='utf8')
        oris_tmp = [' '.join(i) for i in oris]
        oris_s = '\n'.join(oris_tmp)
        wf.write(oris_s)
        wf.close()


    def forward(self, sel, pb, predicted_out, golden_out, mask_id, stc_length_out, sudo_golden_out, sudo_golden_out_1, ori_words, ori_words_length):
        # ls = 0
        # cnt = 0
        #
        # stc_length = sel.shape[1]
        # sel = sel.detach().cpu().numpy()
        # golden_out = golden_out.cpu().numpy()
        #
        # batch = sel.shape[0]
        # bleus = []
        # for i in range(batch):  #batch
        #     bleu = get_reward(predicted_out[i], golden_out[i], stc_length_out[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
        #     bleus.append(bleu)
        # bleus = np.asarray(bleus)
        #
        # wgt = np.asarray([1 for _ in range(batch)])
        # for j in range(stc_length):
        #     ls += (- pb[:, j] *
        #            torch.from_numpy(bleus - self.bl).float().cuda() *
        #            torch.from_numpy(wgt.astype(float)).float().cuda()).sum()
        #     cnt += np.sum(wgt)
        #
        #     wgt = wgt.__and__(sel[:, j] != mask_id)  # vocab_size + 1
        #
        # ls /= cnt
        #
        # bleu = np.average(bleus)
        # self.bl = (self.bl * self.bn + bleu) / (self.bn + 1)
        # self.bn += 1

        ####1####
        ls1 = 0
        cnt1 = 0
        stc_length_seq = sel.shape[1]
        batch = sel.shape[0]
        rewards = []
        for i in range(batch):  #batch
            reward = self.get_reward(predicted_out[i], sudo_golden_out[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards.append(reward)
        rewards = np.asarray(rewards)
        for j in range(stc_length_seq):
            wgt1 = np.asarray([1 if j < stc_length_out[i] else 0 for i in range(batch)])
            # ls1 += (- pb[:, j] *
            #         torch.from_numpy(rewards).float().cuda() *
            #         torch.from_numpy(wgt1.astype(float)).float().cuda()).sum()
            ls1 += (- pb[:, j] *
                    torch.from_numpy(rewards).float().to(self.device) *
                    torch.from_numpy(wgt1.astype(float)).float().to(self.device)).sum()
            cnt1 += np.sum(wgt1)

        ls1 /= cnt1
        rewards_ave1 = np.average(rewards)

        # ####2####
        ls2 = 0
        cnt2 = 0
        stc_length_seq = sel.shape[1]
        batch = sel.shape[0]
        rewards = []
        for i in range(batch):  #batch
            reward = self.get_reward(predicted_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards.append(reward)
        rewards = np.asarray(rewards)

        for j in range(stc_length_seq):
            wgt2 = np.asarray([1 if j < stc_length_out[i] else 0 for i in range(batch)])
            # ls2 += (- pb[:, j] *
            #         torch.from_numpy(rewards).float().cuda() *
            #         torch.from_numpy(wgt2.astype(float)).float().cuda()).sum()
            ls2 += (- pb[:, j] *
                    torch.from_numpy(rewards).float().to(self.device) *
                    torch.from_numpy(wgt2.astype(float)).float().to(self.device)).sum()
            cnt2 += np.sum(wgt2)

        ls2 /= cnt2
        rewards_ave2 = np.average(rewards)

        ####3#####add meaning_preservation as reward
        ls3 = 0
        cnt3 = 0
        stc_length_seq = sel.shape[1]
        batch = sel.shape[0]
        self.write_text(ori_words, ori_words_length, sel, stc_length_out)
        os.system('/home/hanwj/anaconda3/envs/bertscore/bin/python seq2seq_rl/get_bert_score.py')
        meaning_preservation = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp.txt')
        rewards = meaning_preservation * 5  # affect more

        for j in range(stc_length_seq):
            wgt3 = np.asarray([1 if j < stc_length_out[i] else 0 for i in range(batch)])
            ls3 += (- pb[:, j] *
                    torch.from_numpy(rewards).float().to(self.device) *
                    torch.from_numpy(wgt3.astype(float)).float().to(self.device)).sum()
            cnt3 += np.sum(wgt3)

        ls3 /= cnt3
        rewards_ave3 = np.average(rewards)


        loss = ls3  #ls1 + ls2 + ls3
        return loss, ls1, ls3, rewards_ave1, rewards_ave3 #loss, ls, ls1, bleu, bleu1