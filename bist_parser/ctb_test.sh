# test
python barchybrid/src/parser.py --predict --outdir ctb_output --test /home/hanwj/PycharmProjects/structure_adv/data/ctb/test.conllu --extrn sskip.chn.50 --model ctb_output/barchybrid.model30 --params ctb_output/params.pickle
# train
#python barchybrid/src/parser.py --dynet-seed 123456789 --outdir ctb_output  --train /home/hanwj/PycharmProjects/structure_adv/data/ctb/train.conllu --dev /home/hanwj/PycharmProjects/structure_adv/data/ctb/test.conllu --epochs 30 --lstmdims 125 --lstmlayers 2 --extrn sskip.chn.50 --bibi-lstm --k 3 --usehead --userl --pembedding 0
