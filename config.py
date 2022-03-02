#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : config.py
author: Ziwei Xuan
email : xuan64@tamu.edu

Set up configurations.
"""

import os
import sys
import datetime
import argparse
import ast
import numpy as np


def str2bool(v):
    return v.lower() in ('true', '1')


def str2list(v):
    v_le = ast.literal_eval(v)
    if type(v_le) in [int, float]:
        return [v_le]
    elif type(v_le) == list:
        return v_le
    else:
        raise ValueError('Wrong input data type!')


parser = argparse.ArgumentParser()

# Experiment arguments
exp_arg = parser.add_argument_group('data')
exp_arg.add_argument(
    '-task', '--task', type=str, default='new',
    help="Choose 'new', 'test', or 'continue', "
         "as whether this is a new experiment or continue or test based on pre-trained model.")
exp_arg.add_argument(
    '-sp', '--save_path', type=str, default='./train_model',
    help='Folder path of the checkpoint and saved model during training.')
exp_arg.add_argument(
    '-cp', '--ckpt_path', type=str, default='./prev_model/model',
    help='File path of the checkpoint and saved model for continuing training.')
exp_arg.add_argument(
    '-tp', '--test_path', type=str, default='./test_model/model',
    help='File path of the pre-trained model to test on.')
exp_arg.add_argument(
    '-rp', '--record_path', type=str, default='./record',
    help='Folder path for instant recording and final recording during training and testing.')
exp_arg.add_argument(
    '-st', '--sweep_target', type=str, default='snr',
    help="Evaluating the model as sweeping channel SNRs. Not support sweeping other features yet.")
exp_arg.add_argument(
    '-snrr', '--snr_range', type=str2list, default=[0, 31, 5],
    help="Evaluating the model as sweeping channel SNRs. Not support sweeping other features yet.")
exp_arg.add_argument(
    '-g', '--gpu', type=str, default='1',
    help="ID's of the allocated GPUs.")


# Data arguments
data_arg = parser.add_argument_group('data')
data_arg.add_argument(
    '-ipd', '--input_dim', type=int, default=2,
    help='Dimension of the source.')
data_arg.add_argument(
    '-opd', '--output_dim', type=int, default=1,
    help='Dimension of the transmitted signal.')
data_arg.add_argument(
    '-src', '--src_dist', type=str, default='normal',
    help='Probabilistic distribution that the source follows. For now only supports normal distribution.')

# Optimization arguments
opt_arg = parser.add_argument_group('opt')
opt_arg.add_argument(
    '-bs', '--bs', type=int, default=51200*2,
    help='Batch size')
opt_arg.add_argument(
    '-bst', '--bs_test', type=int, default=51200*2,
    help='Batch size for testing.')
opt_arg.add_argument(
    '-lr', '--lr', type=float, default=0.01,
    help='Learning rate')
opt_arg.add_argument(
    '-e', '--epoch', type=int, default=25000,
    help='Number of epochs')
opt_arg.add_argument(
    '-iter', '--iter', type=int, default=100,
    help='Number of iterations per epoch at training')
opt_arg.add_argument(
    '-itv', '--iter_val', type=int, default=100,
    help='Number of iterations per epoch at validation')
opt_arg.add_argument(
    '-itt', '--iter_test', type=int, default=100,
    help='Number of iterations per epoch at testing')
opt_arg.add_argument(
    '-opt', '--optimizer', type=str, default='Adam',
    help='Optimizer used in training')
opt_arg.add_argument(
    '-lrs', '--lr_schedule', type=str, default='WRCosine',
    help='Learning rate schedule: WRCosine, StaircaseDecay, or None.')
opt_arg.add_argument(
    '-sgdm', '--sgd_momentum', type=float, default=0.99,
    help='Momentum of SGD.')
opt_arg.add_argument(
    '-sgdn', '--sgd_nestrov', type=str2bool, default=False,
    help='Whether to use Nestrov in SGD.')
opt_arg.add_argument(
    '-fed', '--first_epoch_decay', type=int, default=4,
    help='The initial epoch time that the learning rate decays to minimum in warm restart cosine annealing.')
opt_arg.add_argument(
    '-mm', '--m_mul', type=float, default=0.9,
    help='The decay factor by which the learning rate attenuates after each period of epochs in warm restart '
         'cosine annealing.')
opt_arg.add_argument(
    '-tm', '--t_mul', type=float, default=1.5,
    help='The expanding factor by which the period of the learning rate grows after each round in warm restart cosine'
         'annealing.')
opt_arg.add_argument(
    '-scdf', '--staircase_decay_factor', type=float, default=0.9,
    help='The decay factor by which the learning rate decays used in staircase decay learning rate schedule.')
opt_arg.add_argument(
    '-sb', '--stair_base', type=str2list, default=[1/2, 3/4],
    help='The portion of the epochs when the learning rate decays in staircase decay learning rate schedule.')
opt_arg.add_argument(
    '-wd', '--weight_decay', type=str2bool, default=False,
    help='Whether to use weight decay or not.')
opt_arg.add_argument(
    '-wdn', '--weight_decay_norm', type=str2bool, default=False,
    help='Whether to use normalized weight decay or not.')
opt_arg.add_argument(
    '-wdf', '--weight_decay_factor', type=float, default=1/320,
    help='The weight decay factor.')
opt_arg.add_argument(
    '-cn', '--clip_norm', type=float, default=5.0,
    help='The number by which the norm of the gradients are clipped.')


# Encoder arguments
enc_arg = parser.add_argument_group('enc')
enc_arg.add_argument(
    '-encn', '--enc_net', type=str, default='RNN',
    help='The network structure choice of the encoder. For now only supports RNN')
enc_arg.add_argument(
    '-encBN', '--enc_BN', type=str2bool, default=True,
    help='Whether the encoder has BN in the encoder.')
enc_arg.add_argument(
    '-encbnm', '--enc_BNm', type=float, default=0.9,
    help='The moment of batch normalization in the encoder.')
enc_arg.add_argument(
    '-encl', '--enc_layer', type=int, default=2,
    help='The number of layers in the encoder.')
enc_arg.add_argument(
    '-ench', '--enc_hidden', type=int, default=16,
    help='The hidden size of the encoder network.')

# Decoder arguments
dec_arg = parser.add_argument_group('dec')
dec_arg.add_argument(
    '-decn', '--dec_net', type=str, default='RNN',
    help='The network structure choice of the decoder. For now only supports RNN')
dec_arg.add_argument(
    '-decBN', '--dec_BN', type=str2bool, default=True,
    help='Whether the encoder has BN in the decoder.')
enc_arg.add_argument(
    '-decbnm', '--dec_BNm', type=float, default=0.9,
    help='The moment of batch normalization in the decoder.')
dec_arg.add_argument(
    '-decl', '--dec_layer', type=int, default=2,
    help='The number of layers in the encoder.')
dec_arg.add_argument(
    '-dech', '--dec_hidden', type=int, default=48,
    help='The hidden size of the encoder network.')

# Channel arguments
channel_arg = parser.add_argument_group('channel')
channel_arg.add_argument(
    '-nmean', '--noise_mean', type=float, default=0.,
    help='Mean of the channel noise.')
channel_arg.add_argument(
    '-snrt', '--snr_train', type=str2list, default=[20.],
    help='SNR of the channel noise.')
channel_arg.add_argument(
    '-pc', '--pc', type=str, default='block-wise',
    help='Power constraint choice before transmission.')
channel_arg.add_argument(
    '-stro', '--snr_train_offset', type=float, default=0.,
    help='The offset to be added in the target channel SNR during training.')
channel_arg.add_argument(
    '-steo', '--snr_test_offset', type=float, default=0.,
    help='The offset to be added in the target channel SNR during training.')


def get_config():
    config, unparsed = parser.parse_known_args()

    # check model configuration
    if config.enc_net not in ['RNN']:
        raise ValueError('No encoder model specified!')
    if config.dec_net not in ['RNN']:
        raise ValueError('No decoder model specified!')

    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)

    if config.task.lower() in ['test']:
        # folder path
        if not os.path.exists(config.test_path + '.meta'):
            raise ValueError(
                "The pre-trained model for testing doesn't exist! Please paste it under {}!".format(config.test_path))
        # set channel SNRs range to be swept during evaluation session
        if len(config.snr_range) != 3:
            raise ValueError('Wrong channel SNR range setting for evaluation!')
        else:
            setattr(config, 'snr_range', np.arange(config.snr_range[0], config.snr_range[1], config.snr_range[2]))

    if config.task.lower() in ['continue']:
        # folder path
        if not os.path.exists(config.ckpt_path + '.meta'):
            raise ValueError(
                "The pre-trained model for testing doesn't exist! Please paste it under {}!".format(config.ckpt_path))

    return config
