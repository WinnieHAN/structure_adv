# Adversarial Attack and Defense of Structured Prediction Models

- Source codes used in EMNLP 2020 paper, *Adversarial Attack and Defense of Structured Prediction Models*.

## Requirements ##

- python 2 & python 3
- anaconda
- pytorch 0.4.1 & pytorch >= 1.0
- transformers
- gensim
- numpy
- bert-score
- pytorch-pretrained-bert
- nltk

## Configuration ##

1. Clone this repository and two anaconda environments: one is python 2 and the other is python 3

2. For python 2 env, install pytorch 0.4.1, gensim, numpy, nltk. <br>
   For python 3 env, install pytorch >= 1.0, transformers, bert-score, pytorch-pretrained-bert and numpy <br>
   You also can download these two enviroments from here. 

3. Downloading sskip embedding and conllu format PTB dataset.
## Dependency Parsing #
1. pretrain parser model. Or you can download our pretrained model here: biaffine, stackptr, bist
```shell script
# pretrain victim model
$ sh examples/run_graphParser.sh

# pretrain reference parser stackPtr
$ sh examples/run_stackPtrParser.sh

# pretrain reference parser bist
$ cd bist_parser
$ sh test.sh
```

2. Move biaffine parser and stackPtr parser to ```./models/parsing/{biaffine, stack_ptr}```. Move pretrained bist parser to ```bist_parser/pretrained/model1```.
3. Pretrain seq2seq sentence generator. Or you also can get our trained seq2sq model here.
```shell script
$ /path/to/python2/envs/python examples/pretrain_seq2seq.py --cuda --mode LSTM \
--num_epochs 30 --batch_size 64 --hidden_size 512 --num_layers 3 --pos_dim 100 --char_dim 100 --num_filters 100 \
--learning_rate 1e-3 --decay_rate 0.05 --schedule 5 --gamma 0.0 \
--p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 \
--word_embedding sskip --word_path data/sskip/sskip.eng.100.gz \
--train data/ptb/train.conllu \
--dev data/ptb/dev.conllu \
--test data/ptb/test.conllu \
--char_embedding random \
--model_path models/parsing/biaffine
```
4. RL training:
```shell script
$ sh examples/run_rl_graph_parser.py
``` 
5. You also can download our trained model here. And run eval to get results report in paper.
```shell script
$ sh examples/eval_rl_graph_parser.py 
```

### POS Tagging
1. Pretrain the victim model.
```shell script
$ sh run_posCRFTagger.sh
```
2. Download and unzip reference parser: [stanford-postagger](http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip), [senna](http://ronan.collobert.com/senna/senna-v3.0.tgz). 
3. Pretrain the seq2seq model.
```shell script
/path/to/python2/envs/python examples/pretrain_seq2seq.py --cuda --mode LSTM \
--num_epochs 30 --batch_size 64 --hidden_size 256 --num_layers 1 --char_dim 30 --num_filters 30 \
--learning_rate 1e-3 --decay_rate 0.05 --schedule 5 --gamma 0.0 \
--p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 \
--word_embedding sskip --word_path data/sskip/sskip.eng.100.gz \
--train data/ptb/train.conllu \
--dev data/ptb/dev.conllu \
--test data/ptb/test.conllu \
--char_embedding random \
--model_path path/to/victim/model/
```
4. RL training.
```shell script
$ sh examples/run_adv_tagger.sh
```
5. You also can download our trained model here. And run eval to get results report in paper.
```shell script
$ sh examples/eval_adv_tagger.py 
```