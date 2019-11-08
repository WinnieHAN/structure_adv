from bert_score import score
import numpy as np
import time
import math

# ===============bertscore===============================

with open("cands.txt") as f:  #cands.txt
    cands = [line.strip() for line in f]
with open("refs.txt") as f:
    refs = [line.strip() for line in f]

# print(cands)
# print(refs)

start = time.time()
# P,R,F = score(cands, refs, bert="bert-base-uncased") # for english
# P,R,F = score(cands, refs, lang="zh", verbose=False)   # for chinese
# P,R,F = score(cands, refs, lang="en", verbose=False)   # for chinese

P,R,F = score(cands, refs, model_type='bert-base-uncased', verbose=False)


np.savetxt('/home/hanwj/PycharmProjects/structure_adv/temp.txt',F.cpu().numpy())
end = time.time()
# print('bert score: ' + str(end-start))
# ===============ppl===============================

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import torch

start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_or_path = '/home/hanwj/PycharmProjects/structure_adv/models/pretrained_model_gpt2'  #'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
model.to(device)
model.eval()
ppls = []
with torch.no_grad():
    for step, s in enumerate(cands):  # actually here is a batch with batchsize=1
        # Put model in training mode.
        if not s:
            ppls.append(0)
            print('space sentence')
            continue
        s = enc.encode(s)  # + [50256]  #50256 is the token_id for <|endoftext|>
        batch = torch.tensor([s]).to(device)
        # print(batch)
        loss = model(batch, lm_labels=batch)  # everage -logp
        # print(loss.cpu().numpy())
        ppls.append(loss.cpu().numpy())  # the small, the better
        # ppls.append(math.exp(loss.cpu().numpy()))  # the small, the better
    # print(ppls)
    np.savetxt('/home/hanwj/PycharmProjects/structure_adv/temp_ppl.txt', np.array(ppls))
end = time.time()
# print('ppl score: ' + str(end-start))