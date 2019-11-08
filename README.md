use python=2.7.13  pytorch=0.4.1 , cuda=10.0
pip install torch==0.4.1 -f https://download.pytorch.org/whl/cu100/stable # CUDA 10.0 build
pip install gensim


python run_rl_graph_parser.py 
--cuda --mode FastLSTM --num_epochs 1000 --batch_size 5 --hidden_size 5 --num_layers 3 --pos_dim 10 --char_dim 10 --num_filters 10 --arc_space 512 --type_space 128 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --objective cross_entropy --decode mst --word_embedding sskip --word_path "data/sskip/sskip.eng.100.gz" --char_embedding random --punctuation '.' '``' "''" ':' ',' --train "data/ptb/train.conllu" --dev "data/ptb/dev.conllu" --test "data/ptb/test.conllu" --model_path "models/parsing/biaffine/" --model_name 'network.pt'



--cuda --mode FastLSTM --num_epochs 1000 --batch_size 32 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --objective cross_entropy --decode mst --word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random --punctuation '.' '``' "''" ':' ',' --train data/ptb/train.conllu --dev data/ptb/dev.conllu --test data/ptb/test.conllu --model_path models/parsing/biaffine/ --model_name network.pt

ctb===== 

1.biaffine parser train

--cuda --mode FastLSTM --num_epochs 1000 --batch_size 32 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --objective cross_entropy --decode mst --word_embedding sskip --word_path data/sskip/sskip.chn.50.gz --char_embedding random --punctuation '（' '）' '，' '。' '“' '”' '：' '、' '《' '》' '；' '——' '—' '－－' '‘' '’' '…' '／' '．' '！' '━━' '〈' '〉' '「' '」' '？' '『' '』' --train data/ctb/train.conllu --dev data/ctb/dev.conllu --test data/ctb/test.conllu --model_path ctb_models/parsing/biaffine/ --model_name network.pt --seq2seq_save_path ctb_models/seq2seq/seq2seq_save_model --network_save_path ctb_models/seq2seq/network_save_model --seq2seq_load_path ctb_models/seq2seq/seq2seq_save_model --network_load_path ctb_models/seq2seq/network_save_model --rl_finetune_seq2seq_save_path ctb_models/rl_finetune/seq2seq_save_model --rl_finetune_network_save_path ctb_models/rl_finetune/network_save_model --rl_finetune_seq2seq_load_path ctb_models/rl_finetune/seq2seq_save_model --rl_finetune_network_load_path ctb_models/rl_finetune/network_save_model

2.biaffine_stack_ptr_train

for graph
'''
python examples/GraphParser.py 
--cuda --mode FastLSTM --num_epochs 1000 --batch_size 32 --hidden_size 512 --num_layers 3 
 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 
 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --pos --char 
 --objective cross_entropy --decode mst 
 --word_embedding sskip --word_path data/sskip/sskip.chn.50.gz --char_embedding random 
 --punctuation '（' '）' '，' '。' '“' '”' '：' '、' '《' '》' '；' '——' '—' '－－' '‘' '’' '…' '／' '．' '！' '━━' '〈' '〉' '「' '」' '？' '『' '』'
--train data/ctb/train.conllu --dev data/ctb/dev.conllu --test data/ctb/test.conllu --model_path ctb_models/parsing/stack_ptr/ --model_name network.pt
'''

for biaffine
'''
examples/StackPointerParser.py --cuda --mode FastLSTM --num_epochs 1000 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 
 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 
 --schedule 20 --double_schedule_decay 5 
 --p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --beam 1 --prior_order inside_out 
 --grandPar --sibling 
 --word_embedding sskip --word_path data/sskip/sskip.chn.50.gz --char_embedding random 
 --punctuation '（' '）' '，' '。' '“' '”' '：' '、' '《' '》' '；' '——' '—' '－－' '‘' '’' '…' '／' '．' '！' '━━' '〈' '〉' '「' '」' '？' '『' '』'
--train data/ctb/train.conllu --dev data/ctb/dev.conllu --test data/ctb/test.conllu --model_path ctb_models/parsing/stack_ptr/ --model_name network.pt
'''

3.bist_parser_train
sh ctb_test.sh

4. change parameters in run_rl_graph_parser.py

=====ptb==================

python examples/run_rl_graph_parser.py --cuda --mode FastLSTM --num_epochs 1000 --batch_size 32 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --objective cross_entropy --decode mst --word_embedding sskip --word_path data/sskip/sskip.eng.100.gz --char_embedding random --punctuation '.' '``' "''" ':' ',' --train data/ptb/train.conllu --dev data/ptb/dev.conllu --test data/ptb/test.conllu --model_path models/parsing/biaffine/ --model_name network.pt --treebank ptb

==environment===========

conda activate bertscore
pip install pytorch_pretrained_bert
conda install scikit-learn
pip uninstall bert-score
git clone https://github.com/Tiiiger/bert_score
cd bert_score
pip install .

--
bert-score/bert_score:
score.py
line 61:        tokenizer = AutoTokenizer.from_pretrained('/home/hanwj/PycharmProjects/check_bertscore_ppl/pretrained_model_bert')  # model_type

utils.py
line 64:        model = AutoModel.from_pretrained('/home/hanwj/PycharmProjects/check_bertscore_ppl/pretrained_model_bert')  # model_type

