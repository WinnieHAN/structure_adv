#!/usr/bin/env bash
  
# cd ./.. | exit

PORT=10040
PREFIX='/home/zhanglw/code/rebase/structure_adv/pos_eval_'

export CUDA_VISIBLE_DEVICES=0
/home/zhanglw/bin/anaconda3/envs/bertscore/bin/python seq2seq_rl/bertscore_ppl_server.py --language_code en --port $PORT --prefix $PREFIX &

/home/zhanglw/bin/anaconda3/envs/advbiaf/bin/python  examples/eval_adv_tagger.py --cuda --mode LSTM \
--hidden_size 256 --num_layers 1 --char_dim 30 --num_filters 30 --tag_space 256 \
--p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --bigram \
--embedding sskip --embedding_dict data/sskip/sskip.eng.100.gz \
--train data/ptb/train.conllu --dev data/ptb/dev.conllu --test data/ptb/test.conllu \
--port $PORT --prefix $PREFIX \
--parserb tagging_models/senna --parserc tagging_models/stanford-postagger \
--rl_finetune_seq2seq_load_path /media/zhanglw/2EF0EF5BF0EF27B3/code/sadv/best_model/tag/seq2seq_save_model_b64_lr5e4_mpe1_noppl_6.pt \
--rl_finetune_network_load_path /media/zhanglw/2EF0EF5BF0EF27B3/code/sadv/best_model/tag/network_save_model_b64_lr5e4_mpe1_noppl_6.pt