#!/usr/bin/env bash

# cd ./.. | exit

PORT=10060
PREFIX='/path/to/any/dir/name'

export CUDA_VISIBLE_DEVICES=0

/path/to/bertscore/envs/python seq2seq_rl/bertscore_ppl_server.py --language_code en --port $PORT --prefix $PREFIX &

/path/to/advbiaf/envs/python examples/eval_rl_graph_parser.py --cuda --mode FastLSTM \
--batch_size 64 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 \
--arc_space 512 --type_space 128 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 \
--word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--train data/ptb/train.conllu --dev data/ptb/dev.conllu --test data/ptb/test.conllu \
--model_path models/parsing/biaffine/ --model_name network.pt \
--rl_finetune_seq2seq_load_path /path/to/seq2seq/model --rl_finetune_network_load_path /path/to/seq2seq/model \
--port $PORT --prefix $PREFIX --unk_weight 500