import codecs

instc = 'data/ctb/test.conllu'
outstc = 'data/ctb/test.conllu'

stcs = codecs.open(instc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
stcs_num = len(token_stcs)
res = []
# wf = codecs.open(outstc, 'w', encoding='utf8')
for i in range(stcs_num):
    print(i)
    line = []
    for j in range(len(token_stcs[i])):  # one sentences
        if token_stcs[i][j][4] == 'PU' and (not token_stcs[i][j][1] in res):
            res.append(token_stcs[i][j][1])

        # wf.write(' '.join(token_stcs[i][j])+'\n')
    # wf.write('\n')
# wf.close()
print(res)