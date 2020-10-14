#!/usr/bin/env bash
  
# cd ./.. | exit

PORT=10045
PREFIX='/path/to/project/root/pos_train_'

export CUDA_VISIBLE_DEVICES=0

/path/to/python3/envs/python seq2seq_rl/bertscore_ppl_server.py --language_code en --port $PORT --prefix $PREFIX &

/path/to/python2/envs/python  examples/run_adv_tagger.py --cuda --mode LSTM --num_epochs 30 \
--batch_size 64 --hidden_size 256 --num_layers 1 --char_dim 30 --num_filters 30 --tag_space 256 --learning_rate 5e-4 \
--decay_rate 0.05 --schedule 5 --gamma 0.0 --dropout std --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --unk_replace 0.0 \
--bigram --embedding sskip --embedding_dict data/sskip/sskip.eng.100.gz \
--train data/ptb/train.conllu --dev data/ptb/dev.conllu --test data/ptb/test.conllu \
--port $PORT --prefix $PREFIX \
--parserb tagging_models/senna --parserc tagging_models/stanford-postagger \
--unk_weight 0 --mp_weight 10 --ppl_weight 0.0001
