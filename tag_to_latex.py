import codecs
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
import math

def write_tree(tokens):
    dic = ['@', '^', '&', '*']
    line = '\n\\begin{dependency}[theme=simple]\n\\begin{deptext}[column sep=0.2mm]\n'
    ws = [i[1] for i in tokens]
    ws = ['/'+i if i in dic else i for i in ws]
    ws = ['$\$$' if i=='$' else i for i in ws]
    ws = ['\#' if i=='#' else i for i in ws]
    ws = ['\#\#' if i=='##' else i for i in ws]
    ws = ['\#\#\#' if i=='###' else i for i in ws]
    ws = ['\#\#\#\#' if i=='###' else i for i in ws]
    ws = ['UNK' if i=='<_UNK>' else i for i in ws]
    ws = ['\%' if i=='%' else i for i in ws]

    line = line + 'ROOT \& ' +' \& '.join(ws)  + ' \\\ \n' \
           + '\end{deptext}\n'
    temp = ''
    for i in range(len(tokens)):
        temp = temp + '\depedge{%s}{%s}{}\n' % (str(int(tokens[i][6])+1), str(i+2))
    line = line + temp + '\n\end{dependency}'
    return line

def write_tag(tokens):
    pos_sent = []
    sent = []
    ws = [i[1] for i in tokens]
    wp = [i[4] for i in tokens]

    cnt = 0
    while True:
        if ws[cnt].startswith('##'):
            res = ws.pop(cnt)
            ws[cnt-1] += res.replace('#', '')
            wp.pop(cnt)
        else:
            cnt += 1
        if cnt >= len(ws):
            break
    for word, pos in zip(ws, wp):
        s = float(len(word))
        s = math.ceil(s / 4.0)
        p = float(len(pos))
        p = math.ceil(p / 4.0)
        l = int(max(s, p))
        sent.append(word + ' '* ((l * 4) - len(word) + 1))
        pos_sent.append(pos + ' '* ((l * 4) - len(pos) + 1))
    line = ''.join(pos_sent)
    line += '\n'
    line += ''.join(sent) + '\n'
    return line

def write_to_tex(tokens, adv_cands, src_cands, i):
    src_ws = src_cands[i]
    line = ''
    line = line + write_tag(adv_cands[i])  # parser A
    line = line + write_tag(tokens)  # golden truth
    line = line + '\nSentence len: %s. Source Sentence: %s' % (str(len(tokens)), src_ws)
    return line

# instc = 'pred_rm_unk.txt'
adv_inparse = '/media/zhanglw/2EF0EF5BF0EF27B3/code/baseline/tag_base/after_taggingpred_test1_parseA.txt_reshape'
src_instc = '/media/zhanglw/2EF0EF5BF0EF27B3/code/baseline/tag_base/tagging_ori_sentences_res.txt'

with open(src_instc) as f:  # cands.txt
    src_cands = [line.strip() for line in f]

stcs = codecs.open(adv_inparse, 'r', encoding='utf8').read().rstrip('\n').lstrip('\n').split('\n\n')
stcs = [i.rstrip('\n').lstrip('\n') for i in stcs]
adv_cands = [[[k for k in line.rstrip('\t').lstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]


dep_parser = CoreNLPDependencyParser(url='http://10.19.124.45:9000')

outstc = 'base_model.tex'
wf = codecs.open(outstc, 'w', encoding='utf8')
with open(src_instc) as f:  # cands.txt
    src_cands = [line.strip() for line in f]

for i in range(0, 50):
    # kk = kk + 1
    # print(kk)
    # if kk>2:
    #     break
    adv_ws = ' '.join([adv_cands[i][ii][1] for ii in range(len(adv_cands[i]))])
    parse, = dep_parser.raw_parse(adv_ws)
    one_stc = parse.to_conll(10)
    stc_tokens = [[k for k in line.rstrip('\t').split('\t')] for line in one_stc.rstrip("\n").rsplit("\n")]

    temp = write_to_tex(stc_tokens, adv_cands, src_cands, i)

    wf.write(temp)
    wf.write('\n\n')
    print(temp)
    # wf.write('\n\n')