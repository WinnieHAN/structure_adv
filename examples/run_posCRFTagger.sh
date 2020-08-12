#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python examples/posCRFTagger.py --cuda --mode LSTM --num_epochs 200 --batch_size 16 --hidden_size 256 --num_layers 1 \
 --char_dim 30 --num_filters 30 --tag_space 256 \
 --learning_rate 0.01 --decay_rate 0.05 --schedule 5 --gamma 0.0 \
 --dropout std --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --unk_replace 0.0 --bigram \
 --embedding sskip --embedding_dict data/sskip/sskip.eng.100.gz \
 --train data/ptb/train.conllu --dev data/ptb/dev.conllu --test data/ptb/test.conllu \
 --load_model_path tagging_models/tagging/rl_finetune/network_save_model_b64_lr5e4_mpe1_noppl_6.pt \
 --model_name tagging_models/tagging/retrain/b64_lr5e4_mpe1_noppl_6