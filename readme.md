use python=2.7.13  pytorch=0.4.1 , cuda=10.0
pip install torch==0.4.1 -f https://download.pytorch.org/whl/cu100/stable # CUDA 10.0 build
pip install gensim


python run_rl_graph_parser.py --cuda --mode FastLSTM --num_epochs 1000 --batch_size 5 --hidden_size 5 --num_layers 3 --pos_dim 10 --char_dim 10 --num_filters 10 --arc_space 512 --type_space 128 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --objective cross_entropy --decode mst --word_embedding sskip --word_path "data/sskip/sskip.eng.100.gz" --char_embedding random --punctuation '.' '``' "''" ':' ',' --train "data/ptb/train.conllu" --dev "data/ptb/dev.conllu" --test "data/ptb/test.conllu" --model_path "models/parsing/biaffine/" --model_name 'network.pt'
