import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

# from bert_score import score
from bert_score.utils import lang2model, model2layers, bert_cos_score_idf
from transformers import AutoModel, AutoTokenizer
import socket

import json
import argparse
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import torch
from collections import defaultdict

bertscore_models = {'en': root_path + '/pretrained_model_bert',
                    'zh': root_path + '/pretrained_model_bert_zh'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pre_bertscore(language_code):
    if language_code not in bertscore_models:
        raise KeyError('Error bert score language code!')
    tokenizer = AutoTokenizer.from_pretrained(bertscore_models[language_code])
    model_type = lang2model[language_code]
    num_layers = model2layers[model_type]
    model = AutoModel.from_pretrained(bertscore_models[language_code])
    model.encoder.layer = \
        torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])

    # not IDF
    idf_dict = defaultdict(lambda: 1.)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    model.to(device)

    return model, tokenizer, idf_dict


def get_bertscore(model, tokenizer, idf_dict, refs, cands):
    with torch.no_grad():
        all_preds = bert_cos_score_idf(model, refs, cands, tokenizer, idf_dict,
                                       verbose=False, device=device,
                                       batch_size=64, all_layers=None)
    P = all_preds[..., 0].cpu()
    R = all_preds[..., 1].cpu()
    F1 = all_preds[..., 2].cpu()
    return P, R, F1


def pre_ppl(language_code):
    if language_code == 'en':
        enc = GPT2Tokenizer.from_pretrained(root_path + '/pretrained_model_gpt2')
        model = GPT2LMHeadModel.from_pretrained(root_path + '/pretrained_model_gpt2')
    elif language_code == 'zh':
        import transformers
        from seq2seq_rl.tokenizations import tokenization_bert
        enc = tokenization_bert.BertTokenizer(vocab_file=root_path + '/pretrained_model_gpt2_zh/vocab.txt')
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(root_path + '/pretrained_model_gpt2_zh')
    model.to(device)
    model.eval()
    return enc, model


def get_ppl(enc, model, cands, language_code):
    ppls = []
    if language_code == 'en':
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
                ppls.append(loss.item())  # the small, the better
    elif language_code == 'zh':
        new_cands = []
        for cand in cands:
            cand = enc.convert_tokens_to_ids(enc.tokenize(cand))
            new_cands.append([enc.convert_tokens_to_ids('[MASK]')] + cand + [enc.convert_tokens_to_ids('[CLS]')])
        ppls = []
        with torch.no_grad():
            for step in new_cands:  # drop last
                #  prepare data
                batch_labels = torch.tensor(step).long().to(device)
                batch_inputs = torch.tensor(step).long().to(device)

                #  forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_labels)
                loss = outputs[0]
                ppls.append(loss.cpu().item())
    return ppls


def main():
    args_parser = argparse.ArgumentParser(description='Get bertscore and ppl score')
    args_parser.add_argument('--port', type=int, required=True, help='socket server')
    args_parser.add_argument('--language_code', type=str, default='en')
    args = args_parser.parse_args()

    port = args.port
    language_code = args.language_code

    # pare bert score model and ppl model
    bs_model, bs_tokenizer, bs_idf_dict = pre_bertscore(language_code)
    ppl_enc, ppl_model = pre_ppl(language_code)

    # prepare socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', port)
    sock.bind(server_address)
    sock.listen(1)

    while True:
        connection, client_address = sock.accept()
        try:
            # get data
            data = connection.recv(1024000)
            if data:
                json_data = json.loads(data)
                refs = json_data['refs']
                cands = json_data['cands']
                P, R, F = get_bertscore(bs_model, bs_tokenizer, bs_idf_dict, refs, cands)
                ppls = get_ppl(ppl_enc, ppl_model, cands, language_code)
                connection.sendall(json.dumps({'bertscore': F.numpy().tolist(),
                                               'ppl': ppls}).encode('utf-8'))
                torch.cuda.empty_cache()
            else:
                break
        finally:
            connection.close()


if __name__ == '__main__':
    # test_bertscore()
    main()
