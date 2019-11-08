from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD, Adamax
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.models import StackPtrNet
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding
import torch.nn as nn

uid = uuid.uuid4().hex[:6]


class third_party_parser(nn.Module):
    def __init__(self, device, word_table, char_table, model_id, args):
        super(third_party_parser, self).__init__()
        # mode = args.mode
        # if model_id==0 and args.treebank == 'ptb':
        if args.treebank == 'ptb':
            model_path = "models/parsing/stack_ptr/"  # args.model_path
        elif args.treebank == 'ctb':
            model_path = "ctb_models/parsing/stack_ptr/"  # args.model_path
        model_name = 'network.pt'  # args.model_name

        model_name = os.path.join(model_path, model_name)

        # data_test = conllx_stacked_data.read_stacked_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, prior_order=prior_order, device=device)

        # save_args()
        arg_path = model_name + '.arg.json'
        # json.dump({'args': arguments, 'kwargs': kwargs}, open(arg_path, 'w'), indent=4)
        [word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window, mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers, num_types, arc_space, type_space] = json.load(open(arg_path, "r"))['args']
        parameters = json.load(open(arg_path, "r"))['kwargs']
        p_in = parameters['p_in']
        p_out = parameters['p_out']
        p_rnn = parameters['p_rnn']
        True = parameters['biaffine']
        use_pos = False  #parameters['pos']
        use_char = False  #parameters['char']
        prior_order = parameters['prior_order']
        skipConnect = parameters['skipConnect']
        grandPar = parameters['grandPar']
        sibling = parameters['sibling']

        window = 3
        self.network = StackPtrNet(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                              mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers,
                              num_types, arc_space, type_space,
                              embedd_word=word_table, embedd_char=char_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                              biaffine=True, pos=use_pos, char=use_char, prior_order=prior_order,
                              skipConnect=skipConnect, grandPar=grandPar, sibling=sibling)
        # if True:
        #     freeze_embedding(network.word_embedd)

        self.network = self.network.to(device)


        ####################
        self.network.load_state_dict(torch.load(model_name))
        self.network = self.network.to(device)
        self.network.eval()

    def parsing(self, word, char, pos, masks, lengths, beam):
        with torch.no_grad():
            heads_pred, types_pred, _, _ = self.network.decode(word, char, pos, mask=masks, length=lengths,
                                                          beam=beam,
                                                          leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)
        return heads_pred, types_pred