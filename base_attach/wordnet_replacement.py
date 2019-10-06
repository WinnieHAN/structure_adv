from nltk.corpus import wordnet as wn
# >> > import nltk
# >> > nltk.download('wordnet')
#   >>> nltk.download('averaged_perceptron_tagger')
import codecs
import numpy as np
from nltk import pos_tag

def word_replace(inf, outf, threshold, threshold_p):
    with open(inf) as f:
        cands = [line.strip() for line in f]

    NNS = ['NN', 'NNS', 'NNP', 'NNPS']
    VVS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    ADV = ['RB', 'RBR', 'RBS']
    ADJ = ['JJ', 'JJR', 'JJS']
    WN2POS = {'v': VVS, 'n': NNS, 'a': ADJ, 'r': ADV}


    wf = codecs.open(outf, 'w', encoding='utf8')
    for si in range(len(cands)):
        replaced_word_num = 0
        stc = cands[si]
        ws = [i.lower() for i in stc.strip().split(' ')]
        ws_pos_tag = pos_tag(ws)
        for wi in range(len(ws)):
            try:
                a = wn.synsets(ws[wi])
            except:
                continue
            cur_tag = ws_pos_tag[wi][1]
            if not cur_tag in NNS+VVS+ADV+ADJ:
                continue
            if not wn.synsets(ws[wi]):
                continue
            if not cur_tag in WN2POS[wn.synsets(ws[wi])[0].pos()]:   # ambig
                continue
            synsets = wn.synsets(ws[wi], pos=wn.synsets(ws[wi])[0].pos())
            for scand in synsets:
                if not scand.name().split('.')[0]==ws[wi]:
                    if not scand.path_similarity(wn.synsets(ws[wi])[0])>threshold:
                        ws[wi] = wn.synsets(ws[wi])[0].name().split('.')[0]
                        replaced_word_num += 1
                        break
            if float(replaced_word_num)/len(ws)>threshold_p:
                break
        wf.write(' '.join(ws))
        wf.write('\n')
    wf.close()


if __name__ == '__main__':
    inf = 'refs.txt'
    outf = 'output_wordnet.txt' #'output.txt'
    word_replace(inf, outf, 0, 0)