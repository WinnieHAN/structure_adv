# test
# python barchybrid/src/parser.py --predict --outdir output --test /home/hanwj/PycharmProjects/NeuroNLP2/data/ptb/test.conllu --extrn sskip.100.vectors --model pretrained/model1/barchybrid.model30 --params pretrained/model1/params.pickle
# train
python barchybrid/src/parser.py --dynet-seed 123456789 --outdir output  --train /home/hanwj/Code/PycharmProjects/NeuroNLP2/data/ptb/train.conllu --dev /home/hanwj/Code/PycharmProjects/NeuroNLP2/data/ptb/test.conllu --epochs 30 --lstmdims 125 --lstmlayers 2 --extrn sskip.100.vectors --bibi-lstm --k 3 --usehead --userl --pembedding 0
