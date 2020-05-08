#!/usr/bin/env bash

# cd ./.. | exit

PORT=10048
PREFIX='/home/zhanglw/code/structure_adv/test_'

export CUDA_VISIBLE_DEVICES=0

# /home/zhanglw/bin/anaconda3/envs/bertscore/bin/python seq2seq_rl/bertscore_ppl_server.py --language_code zh --port $PORT --prefix $PREFIX &
#
#/home/zhanglw/bin/anaconda3/envs/advbiaf/bin/python examples/run_rl_graph_parser.py --cuda --mode FastLSTM --num_epochs 80 --batch_size 10 --hidden_size 512 \
#--num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 --opt adam \
#--learning_rate 0.001 --decay_rate 0.97 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 --p_in 0.33 --p_rnn 0.33 0.33 \
#--p_out 0.33 --unk_replace 0.5 --objective cross_entropy --decode mst --word_embedding sskip \
#--word_path data/sskip/sskip.chn.50.gz --char_embedding random --treebank ctb \
#--punctuation '（' '）' '，' '。' '“' '”' '：' '、' '《' '》' '；' '——' '—' '－－' '‘' '’' '…' '／' '．' '！' '━━' '〈' '〉' '「' '」' '？' '『' '』' \
#--train data/ctb/train.conllu --dev data/ctb/dev.conllu --test data/ctb/test.conllu \
#--model_path ctb_models/parsing/biaffine/ --model_name network.pt \
#--seq2seq_save_path ctb_models/seq2seq/seq2seq_save_model --network_save_path ctb_models/seq2seq/network_save_model \
#--seq2seq_load_path ctb_models/seq2seq/seq2seq_save_model --network_load_path ctb_models/seq2seq/network_save_model \
#--rl_finetune_seq2seq_save_path ctb_models/rl_finetune/seq2seq_save_model --rl_finetune_network_save_path ctb_models/rl_finetune/network_save_model \
#--rl_finetune_seq2seq_load_path ctb_models/rl_finetune/seq2seq_save_model --rl_finetune_network_load_path ctb_models/rl_finetune/network_save_model \
#--port $PORT --prefix $PREFIX

/home/zhanglw/bin/anaconda3/envs/bertscore/bin/python seq2seq_rl/bertscore_ppl_server.py --language_code en --port $PORT --prefix $PREFIX &

/home/zhanglw/bin/anaconda3/envs/advbiaf/bin/python examples/run_rl_graph_parser.py --cuda --mode FastLSTM \
--num_epochs 1000 --batch_size 32 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 \
--arc_space 512 --type_space 128 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 \
--gamma 0.0 --clip 5.0 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --objective cross_entropy \
--decode mst --word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--train data/ptb/train.conllu --dev data/ptb/dev.conllu --test data/ptb/test.conllu \
--model_path models/parsing/biaffine/ --model_name network.pt --treebank ptb \
--model_path "models/parsing/biaffine/" --model_name 'network.pt' \
--seq2seq_save_path models/seq2seq/seq2seq_save_model --network_save_path models/seq2seq/network_save_model \
--rl_finetune_seq2seq_save_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_save_path models/rl_finetune/network_save_model \
--rl_finetune_seq2seq_load_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_load_path models/rl_finetune/network_save_model \
--port $PORT --prefix $PREFIX
