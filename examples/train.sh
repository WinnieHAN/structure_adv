/home/zhanglw/bin/anaconda3/envs/advbiaf/bin/python examples/training_set_generator.py --cuda --mode FastLSTM \
--num_epochs 1000 --batch_size 64 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 \
--arc_space 512 --type_space 128  --unk_replace 0.5 --objective cross_entropy \
--decode mst --word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random \
--train data/ptb/billp --dev data/ptb/billp_bc_same.conllu --test data/ptb/test.conllu \
--model_path models/parsing/biaffine/ --model_name network.pt --treebank ptb \
--seq2seq_save_path models/seq2seq/seq2seq_save_model --network_save_path models/seq2seq/network_save_model \
--seq2seq_load_path models/seq2seq/seq2seq_save_model --network_load_path models/seq2seq/network_save_model \
--rl_finetune_seq2seq_save_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_save_path models/rl_finetune/network_save_model \
--rl_finetune_seq2seq_load_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_load_path models/rl_finetune/network_save_model \
--prefix _billp_1

/home/zhanglw/bin/anaconda3/envs/advbiaf/bin/python examples/training_set_generator.py --cuda --mode FastLSTM \
--num_epochs 1000 --batch_size 64 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 \
--arc_space 512 --type_space 128  --unk_replace 0.5 --objective cross_entropy \
--decode mst --word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random \
--train data/ptb/billp --dev data/ptb/billp_bc_same.conllu --test data/ptb/test.conllu \
--model_path models/parsing/biaffine/ --model_name network.pt --treebank ptb \
--seq2seq_save_path models/seq2seq/seq2seq_save_model --network_save_path models/seq2seq/network_save_model \
--seq2seq_load_path models/seq2seq/seq2seq_save_model --network_load_path models/seq2seq/network_save_model \
--rl_finetune_seq2seq_save_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_save_path models/rl_finetune/network_save_model \
--rl_finetune_seq2seq_load_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_load_path models/rl_finetune/network_save_model \
--prefix _billp_2

/home/zhanglw/bin/anaconda3/envs/advbiaf/bin/python examples/training_set_generator.py --cuda --mode FastLSTM \
--num_epochs 1000 --batch_size 64 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 \
--arc_space 512 --type_space 128  --unk_replace 0.5 --objective cross_entropy \
--decode mst --word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random \
--train data/ptb/billp --dev data/ptb/billp_bc_same.conllu --test data/ptb/test.conllu \
--model_path models/parsing/biaffine/ --model_name network.pt --treebank ptb \
--seq2seq_save_path models/seq2seq/seq2seq_save_model --network_save_path models/seq2seq/network_save_model \
--seq2seq_load_path models/seq2seq/seq2seq_save_model --network_load_path models/seq2seq/network_save_model \
--rl_finetune_seq2seq_save_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_save_path models/rl_finetune/network_save_model \
--rl_finetune_seq2seq_load_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_load_path models/rl_finetune/network_save_model \
--prefix _billp_3

/home/zhanglw/bin/anaconda3/envs/advbiaf/bin/python examples/training_set_generator.py --cuda --mode FastLSTM \
--num_epochs 1000 --batch_size 64 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 \
--arc_space 512 --type_space 128  --unk_replace 0.5 --objective cross_entropy \
--decode mst --word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random \
--train data/ptb/billp --dev data/ptb/billp_bc_same.conllu --test data/ptb/test.conllu \
--model_path models/parsing/biaffine/ --model_name network.pt --treebank ptb \
--seq2seq_save_path models/seq2seq/seq2seq_save_model --network_save_path models/seq2seq/network_save_model \
--seq2seq_load_path models/seq2seq/seq2seq_save_model --network_load_path models/seq2seq/network_save_model \
--rl_finetune_seq2seq_save_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_save_path models/rl_finetune/network_save_model \
--rl_finetune_seq2seq_load_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_load_path models/rl_finetune/network_save_model \
--prefix _billp_4

/home/zhanglw/bin/anaconda3/envs/advbiaf/bin/python examples/training_set_generator.py --cuda --mode FastLSTM \
--num_epochs 1000 --batch_size 64 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 \
--arc_space 512 --type_space 128  --unk_replace 0.5 --objective cross_entropy \
--decode mst --word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random \
--train data/ptb/billp --dev data/ptb/billp_bc_same.conllu --test data/ptb/test.conllu \
--model_path models/parsing/biaffine/ --model_name network.pt --treebank ptb \
--seq2seq_save_path models/seq2seq/seq2seq_save_model --network_save_path models/seq2seq/network_save_model \
--seq2seq_load_path models/seq2seq/seq2seq_save_model --network_load_path models/seq2seq/network_save_model \
--rl_finetune_seq2seq_save_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_save_path models/rl_finetune/network_save_model \
--rl_finetune_seq2seq_load_path models/rl_finetune/seq2seq_save_model --rl_finetune_network_load_path models/rl_finetune/network_save_model \
--prefix _billp_5