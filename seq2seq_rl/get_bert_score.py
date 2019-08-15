from bert_score import score
import numpy
import sys


with open("cands.txt") as f:
    cands = [line.strip() for line in f]
with open("refs.txt") as f:
    refs = [line.strip() for line in f]

# print(cands)
# print(refs)

P,R,F = score(cands, refs, bert="bert-base-uncased")

numpy.savetxt('/home/hanwj/PycharmProjects/structure_adv/temp.txt',F.cpu().numpy())
