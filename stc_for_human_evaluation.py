import codecs
import random



def stc_select(selected_idx, ins, ins_s):
    wf = codecs.open(ins_s, 'w', encoding='utf8')

    stcs = codecs.open(ins, 'r', encoding='utf8').read().rstrip('\n').lstrip('\n').split('\n\n')
    stcs = [i.rstrip('\n').lstrip('\n') for i in stcs]
    token_stcs = [[[k for k in line.rstrip('\t').lstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]

    for i in selected_idx:
        print(i)
        if not token_stcs[i] == [['']]:
            for j in range(len(token_stcs[i])):  # one sentences
                line = token_stcs[i][j][0] + '\t' + token_stcs[i][j][1] + '\t' + token_stcs[i][j][2] + '\t' + \
                       token_stcs[i][j][3] + '\t' + \
                       token_stcs[i][j][4] + '\t' + token_stcs[i][j][5] + \
                       '\t' + token_stcs[i][j][6] + '\t' + token_stcs[i][j][7] + '\n'
                wf.write(line)
            wf.write('\n')
    wf.close()


def stc_seq_select(selected_idx, ins, ins_s):
    wf = codecs.open(ins_s, 'w', encoding='utf8')

    stcs = codecs.open(ins, 'r', encoding='utf8').read().rstrip('\n').lstrip('\n').split('\n')

    for i in selected_idx:
        wf.write(stcs[i])
        wf.write('\n')
    wf.close()

if __name__ == '__main__':
    random.seed(12345)
    selected_idx = random.sample(range(600), 50)
    print(selected_idx)

    instcA = './dumped/pred_test0_parseA.txt'
    instcB = './dumped/pred_test0_parseB.txt'
    instcC = './dumped/pred_test0_parseC.txt'
    srcstc = './dumped/src_test0'

    instcA_s = './dumped/for_human_eval/pred_test0_parseA.txt'
    instcB_s = './dumped/for_human_eval/pred_test0_parseB.txt'
    instcC_s = './dumped/for_human_eval/pred_test0_parseC.txt'
    srcstc_s = './dumped/for_human_eval/src_test0'

    stc_select(selected_idx, instcA, instcA_s)
    stc_select(selected_idx, instcB, instcB_s)
    stc_select(selected_idx, instcC, instcC_s)
    stc_seq_select(selected_idx, srcstc, srcstc_s)
