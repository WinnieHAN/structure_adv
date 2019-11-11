from nltk.translate.bleu_score import sentence_bleu as BLEU
import numpy as np
import torch.nn as nn
import torch, os, codecs, math
# ref = [[1, 2, 3, 4, 5, 6]]
# cnd = [1, 3, 4, 5, 6]
# bleu = BLEU(ref, cnd)
#
# print('BLEU: %.4f%%' % (bleu * 100))

#TODO(lwzhang) temporarily used hard code to sepcecify saved score file
SCORE_PREFIX = 'task1_'

def get_bleu(out, dec_out, vocab_size):
    out = out.tolist()
    dec_out = dec_out.tolist()
    stop_token = 1
    if stop_token in out:
        cnd = out[:out.index(stop_token)]
    else:
        cnd = out

    if stop_token in dec_out:
        ref = [dec_out[:dec_out.index(stop_token)]]
    else:
        ref = [dec_out]

    bleu = BLEU(ref, cnd)

    return bleu


def get_correct(out, dec_out, num_words):
    out = out.tolist()
    dec_out = dec_out.tolist()
    stop_token = 1
    if stop_token in out:
        cnd = out[:out.index(stop_token)]
    else:
        cnd = out

    if stop_token in dec_out:
        ref = [dec_out[:dec_out.index(stop_token)]]
    else:
        ref = [dec_out]
    tmp = [1 if cnd[i] == ref[i] else 0 for i in range(1, min(len(cnd), len(ref)))]
    if not tmp:
        stc_crt = 0
    else:
        stc_crt = sum(tmp)
    assert max(len(cnd), len(ref)) - 1>0
    return stc_crt, max(len(cnd), len(ref))-1


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
            stop_token = 1
            wgt = wgt.__and__(sel[:, j] != stop_token)  # vocab_size + 1

        ls /= cnt

        bleu = np.average(bleus)
        self.bl = (self.bl * self.bn + bleu) / (self.bn + 1)
        self.bn += 1

        return ls


class LossBiafRL1(nn.Module):
    def __init__(self, device, word_alphabet, vocab_size):
        super(LossBiafRL1, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet
        self.vocab_size = vocab_size

    def get_reward(self, out, dec_out, length_out, ori_words, ori_words_length):
        # out = out.tolist()
        # dec_out = dec_out.tolist()
        # word_alphabet.get_instance(one_word).encode('utf-8')
        # reward = 0
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
        meaning_preservation = np.loadtxt('temp.txt')
        rewards = meaning_preservation  # affect more

        bleus_w = []
        for i in range(batch):
            bleu = get_bleu(ori_words[i], sel[i], self.vocab_size)

            bleus_w.append(bleu)
        bleus_w = np.asarray(bleus_w)
        # rewards = rewards + bleus_w
        rewards = bleus_w * 100  # 8.26

        for j in range(stc_length_seq):
            wgt3 = np.asarray([1 if j < min(stc_length_out[i]+1, stc_length_seq) else 0 for i in range(batch)])  # consider in STOP token
            ls3 += (- pb[:, j] *
                    torch.from_numpy(rewards).float().to(self.device) *
                    torch.from_numpy(wgt3.astype(float)).float().to(self.device)).sum()
            cnt3 += np.sum(wgt3)

        ls3 /= cnt3
        rewards_ave3 = np.average(rewards)


        loss = ls3
        # loss = ls1
        return loss, ls1, ls3, rewards_ave1, rewards_ave3 #loss, ls, ls1, bleu, bleu1



class LossBiafRL(nn.Module):
    def __init__(self, device, word_alphabet, vocab_size):
        super(LossBiafRL, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet
        self.vocab_size = vocab_size

    def get_reward_diff(self, out, dec_out, length_out, ori_words, ori_words_length):
        stc_dda = sum([0 if out[i] == dec_out[i] else 1 for i in range(1, length_out)])

        reward = stc_dda

        return reward

    def get_reward_same(self, out, dec_out, length_out, ori_words, ori_words_length):
        stc_dda = sum([1 if out[i] == dec_out[i] else 0 for i in range(1, length_out)])

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
        ####1####
        batch = sel.shape[0]
        rewards_z1 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
        rewards_z1 = np.asarray(rewards_z1)

        #####2####
        batch = sel.shape[0]
        rewards_z2 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z2.append(reward)
        rewards_z2 = np.asarray(rewards_z2)

        #####3####
        batch = sel.shape[0]
        rewards_z3 = []
        for i in range(batch):  #batch
            reward = self.get_reward_same(sudo_golden_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z3.append(reward)
        rewards_z3 = np.asarray(rewards_z3)

        ####3#####add meaning_preservation as reward
        batch = sel.shape[0]
        self.write_text(ori_words, ori_words_length, sel, stc_length_out)
        os.system('/home/hanwj/anaconda3/envs/bertscore/bin/python seq2seq_rl/get_bertscore_ppl.py')
        meaning_preservation = np.loadtxt(SCORE_PREFIX + 'temp.txt')*100
        logppl = np.loadtxt(SCORE_PREFIX + 'temp_ppl.txt') # * (-0.1)
        ppl = -np.exp(logppl) * 0.001
        # rewards = meaning_preservation * 10  # affect more

        bleus_w = []
        for i in range(batch):
            bleu = get_bleu(ori_words[i], sel[i], self.vocab_size)

            bleus_w.append(bleu)
        bleus_w = np.asarray(bleus_w)

        #-----------------------------------------------

        rewards = (meaning_preservation + ppl + rewards_z1 + rewards_z2 + rewards_z3)*0.001      #TODO  0.1# + bleus_w*5
        # rewards = bleus_w * 10  # 8.26

        ls3 = 0
        cnt3 = 0
        stc_length_seq = sel.shape[1]
        for j in range(stc_length_seq):
            wgt3 = np.asarray([1 if j < min(stc_length_out[i]+1, stc_length_seq) else 0 for i in range(batch)])  # consider in STOP token
            ls3 += (- pb[:, j] *
                    torch.from_numpy(rewards-self.bl).float().to(self.device) *  # rewards-self.bl
                    torch.from_numpy(wgt3.astype(float)).float().to(self.device)).sum()
            cnt3 += np.sum(wgt3)

        ls3 /= cnt3
        rewards_ave3 = np.average(rewards)
        self.bl = (self.bl * self.bn + rewards_ave3) / (self.bn + 1)
        self.bn += 1


        loss = ls3

        # print('rewards_z1: ', np.average(rewards_z1))
        # print('rewards_z2: ', np.average(rewards_z2))
        # print('meaning_preservation: ', np.average(meaning_preservation))
        # print('ppl: ', np.average(ppl))

        # loss = ls1
        return loss, np.average(rewards_z1), np.average(rewards_z2), np.average(rewards_z3), np.average(meaning_preservation), np.average(ppl) #loss, ls, ls1, bleu, bleu1