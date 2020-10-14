#!/usr/bin/env bash
  
# cd ./.. | exit

PORT=10040
PREFIX='/path/to/project/root/pos_eval_'

export CUDA_VISIBLE_DEVICES=0
/path/to/python3/envs/python seq2seq_rl/bertscore_ppl_server.py --language_code en --port $PORT --prefix $PREFIX &

/path/to/python3/envs/python  examples/eval_adv_tagger.py --cuda --mode LSTM \
--hidden_size 256 --num_layers 1 --char_dim 30 --num_filters 30 --tag_space 256 \
--p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --bigram \
--embedding sskip --embedding_dict data/sskip/sskip.eng.100.gz \
--train data/ptb/train.conllu --dev data/ptb/dev.conllu --test data/ptb/test.conllu \
--port $PORT --prefix $PREFIX \
--parserb tagging_models/senna --parserc tagging_models/stanford-postagger \
--rl_finetune_seq2seq_load_path /path/to/seq2seq/model/model_name.pt \
--rl_finetune_network_load_path /path/to/network/model/model_name.pt