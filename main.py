#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : main.py
author: Ziwei Xuan
email : xuan64@tamu.edu

Main script. Start running model from main.py.
"""

from utils.model import *
from utils.train import *
from config import *

def main():
    config = get_config()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)  # model will be trained on designated GPU

    print('Experiment with {} source, with BW k = {}, n = {}.'.format(config.src_dist, config.input_dim,
                                                                      config.output_dim))
    if config.task in ['new', 'New', 'NEW', 'continue', 'Continue', 'CONTINUE']:
        for snr in config.snr_train:
            # set channel noise variance
            setattr(config, 'snr', snr)
            tf.reset_default_graph()
            AJSCC(config)
    elif config.task in ['test', 'Test', 'TEST']:
        tf.reset_default_graph()
        AJSCC(config)
    else:
        raise ValueError('Wrong task value!')
    # end of main


if __name__ == "__main__":
    main()
