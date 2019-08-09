## AttributeError: module 'dynet' has no attribute 'ParameterCollection'

https://dynet.readthedocs.io/en/latest/python_saving_tutorial.html
Using m2.populate("/tmp/tmp.model") to replece load()

install Dynet of CPU version:  pip install dynet


PyTorch implementation of the BIST Parsers (for graph based parser only): https://github.com/wddabc/bist-parser.git

# Result 
python barchybrid/src/parser.py --predict --outdir output --test /home/wenjuan/PycharmProjects/NeuroNLP2/data/ptb/test.conllu --extrn sskip.100.vectors --model pretrained/model1/barchybrid.model30 --params pretrained/model1/params.pickle 

python barchybrid/src/parser.py --dynet-seed 123456789 --outdir output --train /home/wenjuan/PycharmProjects/NeuroNLP2/data/ptb/train.conllu --dev /home/wenjuan/PycharmProjects/NeuroNLP2/data/ptb/test.conllu --epochs 30 --lstmdims 125 --lstmlayers 2 --extrn sskip.100.vectors --bibi-lstm --k 3 --usehead --userl --pembedding 0

Metrics    | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |    100.00 |    100.00 |    100.00 |    100.00
XPOS       |    100.00 |    100.00 |    100.00 |    100.00
Feats      |    100.00 |    100.00 |    100.00 |    100.00
AllTags    |    100.00 |    100.00 |    100.00 |    100.00
Lemmas     |    100.00 |    100.00 |    100.00 |    100.00
UAS        |     91.70 |     91.70 |     91.70 |     91.70
LAS        |     89.63 |     89.63 |     89.63 |     89.63
WeightedLAS|     89.24 |     89.31 |     89.27 |     89.31


# Dependency Parsing on Penn Treebank

https://paperswithcode.com/sota/dependency-parsing-on-penn-treebank

# Dynet
 Trainer::update_epoch has been deprecated and doesn't do anything. Please remove it from your code, and control the learning rate of the trainer directly, for example by: 'trainer.learning_rate /= (1 - rate_decay)', see https://github.com/clab/dynet/pull/695 for details.

Rate Decay Before

// At beginning of training
Trainer trainer(initial_learning_rate, rate_decay)
// After every epoch
trainer.update_epoch()

Rate Decay After

// At beginning of training
Trainer trainer(initial_learning_rate)
// After every epoch
trainer.learning_rate /= (1 - rate_decay)


# BIST Parsers
## Graph & Transition based dependency parsers using BiLSTM feature extractors.

The techniques behind the parser are described in the paper [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198). Futher materials could be found [here](http://elki.cc/#/article/Simple%20and%20Accurate%20Dependency%20Parsing%20Using%20Bidirectional%20LSTM%20Feature%20Representations).

#### Required software

 * Python 2.7 interpreter
 * [DyNet library](https://github.com/clab/dynet/tree/master/python)

#### Train a parsing model

The software requires having a `training.conll` and `development.conll` files formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat).
For the faster graph-based parser change directory to `bmstparser` (1200 words/sec), and for the more accurate transition-based parser change directory to `barchybrid` (800 word/sec). The benchmark was performed on a Mac book pro with i7 processor. The graph-based parser acheives an accuracy of 93.8 UAS and the transition-based parser an accuracy of 94.7 UAS on the standard Penn Treebank dataset (Standford Dependencies). The transition-based parser requires no part-of-speech tagging and setting all the tags to NN will produce the expected accuracy. The model and param files achieving those scores are available for download ([Graph-based model](https://www.dropbox.com/sh/v9cbshnmb36km6v/AADgBS9hb9vy0o-UBZW9AbbKa/bestfirstorder.tar.gz?dl=0), [Transition-based model](https://www.dropbox.com/sh/v9cbshnmb36km6v/AACEPp3DLQeJnRA_QyPmll93a/bestarchybrid.tar.gz?dl=0)). The trained models include improvements beyond those described in the paper, to be published soon.

To train a parsing model with for either parsing architecture type the following at the command prompt:

    python src/parser.py --dynet-seed 123456789 [--dynet-mem XXXX] --outdir [results directory] --train training.conll --dev development.conll --epochs 30 --lstmdims 125 --lstmlayers 2 [--extrn extrn.vectors] --bibi-lstm

We use the same external embedding used in [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://arxiv.org/abs/1505.08075) which can be downloaded from the authors [github repository](https://github.com/clab/lstm-parser/) and [directly here](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).

If you are training a transition-based parser then for optimal results you should add the following to the command prompt `--k 3 --usehead --userl`. These switch will set the stack to 3 elements; use the BiLSTM of the head of trees on the stack as feature vectors; and add the BiLSTM of the right/leftmost children to the feature vectors.

Note 1: You can run it without pos embeddings by setting the pos embedding dimensions to zero (--pembedding 0).

Note 2: The reported test result is the one matching the highest development score.

Note 3: The parser calculates (after each iteration) the accuracies excluding punctuation symbols by running the `eval.pl` script from the CoNLL-X Shared Task and stores the results in directory specified by the `--outdir`.

Note 4: The external embeddings parameter is optional and better not used when train/predicting a graph-based model.

#### Parse data with your parsing model

The command for parsing a `test.conll` file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat) with a previously trained model is:

    python src/parser.py --predict --outdir [results directory] --test test.conll [--extrn extrn.vectors] --model [trained model file] --params [param file generate during training]

The parser will store the resulting conll file in the out directory (`--outdir`).

Note 1: If you are using the arc-hybrid trained model we provided please use the `--extrn` flag and specify the location of the external embeddings file.

Note 2: If you are using the first-order trained model we provided please do not use the `--extrn` flag.

#### Citation

If you make use of this software for research purposes, we'll appreciate citing the following:

    @article{DBLP:journals/tacl/KiperwasserG16,
        author    = {Eliyahu Kiperwasser and Yoav Goldberg},
        title     = {Simple and Accurate Dependency Parsing Using Bidirectional {LSTM}
               Feature Representations},
        journal   = {{TACL}},
        volume    = {4},
        pages     = {313--327},
        year      = {2016},
        url       = {https://transacl.org/ojs/index.php/tacl/article/view/885},
        timestamp = {Tue, 09 Aug 2016 14:51:09 +0200},
        biburl    = {http://dblp.uni-trier.de/rec/bib/journals/tacl/KiperwasserG16},
        bibsource = {dblp computer science bibliography, http://dblp.org}
    }
    
#### Forks

[BIST-PyTorch](https://github.com/wddabc/bist-parser): A PyTorch implementation of the BIST Parsers (for graph based parser only). 

[BIST-COVINGTON](https://github.com/aghie/LyS-FASTPARSE): A neural implementation of the Covington's algorithm for non-projective dependency parsing. It extends the original BIST transition-based a greedy parser by including a dynamic oracle for non-projective parsing to mitigate error propagation.

[Uppsala Parser](https://github.com/UppsalaNLP/uuparser):  A transition-based parser for Universal Dependencies with BiLSTM word and character representations. 

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact elikip@gmail.com

#### Credits

[Eliyahu Kiperwasser](http://elki.cc)

[Yoav Goldberg](https://www.cs.bgu.ac.il/~yoavg/uni/)

