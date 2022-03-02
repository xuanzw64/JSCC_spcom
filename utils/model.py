#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : model.py
author: Ziwei Xuan
email : xuan64@tamu.edu

Encoder and decoder model.
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl


class Encoder(tfkl.Layer):
    """Encoder model"""
    def __init__(self, config, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.net = config.enc_net
        self.hidden_unit = config.enc_hidden
        self.layers = config.enc_layer
        self.output_dim = config.output_dim
        self.input_dim = config.input_dim
        self.bn_choice = config.enc_BN
        self.momentum = config.enc_BNm

    def call(self, x, training=True):
        if self.net == 'RNN':
            if self.input_dim >= self.output_dim:
                # bandwidth compression
                enc_out = self.rnn_compress(x, training)
            else:
                # bandwidth expansion
                enc_out = self.rnn_expand(x, training)
        else:
            raise ValueError('Not supporting this network type in encoder!')
        return enc_out

    def rnn_compress(self, x, training=True):
        x_reshape = tf.reshape(x, [-1, self.output_dim, int(self.input_dim / self.output_dim)])
        rnn_in = x_reshape
        for _ in range(self.layers):
            rnn_out = tfkl.Bidirectional(tfkl.CuDNNLSTM(self.hidden_unit, return_sequences=True))(rnn_in)
            rnn_in = tf.layers.batch_normalization(rnn_out, training=training,
                                                   momentum=self.momentum) if self.bn_choice else rnn_out
        td = tfkl.TimeDistributed(tfkl.Dense(1))(rnn_in)
        td_reshape = tf.reshape(td, [-1, self.output_dim])
        return td_reshape

    def rnn_expand(self, x, training=True):
        x_reshape = tf.reshape(x, [-1, self.input_dim, 1])
        rnn_in = x_reshape
        for _ in range(self.layers):
            rnn_out = tfkl.Bidirectional(tfkl.CuDNNLSTM(self.hidden_unit, return_sequences=True))(rnn_in)
            rnn_in = tf.layers.batch_normalization(rnn_out, training=training,
                                                   momentum=self.momentum) if self.bn_choice else rnn_out
        td = tfkl.TimeDistributed(tfkl.Dense(int(self.output_dim / self.input_dim)))(rnn_in)
        td_reshape = tf.reshape(td, [-1, self.output_dim])

        return td_reshape


class Decoder(tfkl.Layer):
    """Decoder model"""
    def __init__(self, config, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.net = config.dec_net
        self.hidden_unit = config.dec_hidden
        self.layers = config.enc_layer
        self.output_dim = config.input_dim
        self.input_dim = config.output_dim
        self.bn_choice = config.dec_BN
        self.momentum = config.dec_BNm

    def call(self, x, training=True):
        if self.net == 'RNN':
            if self.output_dim >= self.input_dim:
                # bandwidth compression
                dec_out = self.rnn_compress(x, training)
            else:
                # bandwidth expansion
                dec_out = self.rnn_expand(x, training)
        else:
            raise ValueError('Not supporting this network type in decoder!')
        return dec_out

    def rnn_compress(self, x, training=True):
        x_reshape = tf.reshape(x, [-1, self.input_dim, 1])
        rnn_in = x_reshape
        for _ in range(self.layers):
            rnn_out = tfkl.Bidirectional(tfkl.CuDNNLSTM(self.hidden_unit, return_sequences=True))(rnn_in)
            rnn_in = tf.layers.batch_normalization(rnn_out, training=training,
                                                   momentum=self.momentum) if self.bn_choice else rnn_out
        td = tfkl.TimeDistributed(tfkl.Dense(int(self.output_dim / self.input_dim)))(rnn_in)
        re = tf.reshape(td, [-1, self.output_dim])

        return re

    def rnn_expand(self, x, training=True):
        x_reshape = tf.reshape(x, [-1, self.output_dim, int(self.input_dim / self.output_dim)])
        rnn_in = x_reshape
        for _ in range(self.layers):
            rnn_out = tfkl.Bidirectional(tfkl.CuDNNLSTM(self.hidden_unit, return_sequences=True))(rnn_in)
            rnn_in = tf.layers.batch_normalization(rnn_out, training=training,
                                                   momentum=self.momentum) if self.bn_choice else rnn_out
        td = tfkl.TimeDistributed(tfkl.Dense(1))(rnn_in)
        re = tf.reshape(td, [-1, self.output_dim])

        return re


def power_constraint(enc_output, config):
    pc_choice = config.pc
    if pc_choice == 'block-wise':
        enc_output_mean, enc_output_var = tf.nn.moments(enc_output, axes=[0, 1])
        tx = (enc_output - tf.cast(enc_output_mean, tf.float32)) / tf.cast(tf.sqrt(enc_output_var), enc_output.dtype)
    elif pc_choice == 'bit-wise':
        enc_output_mean, enc_output_var = tf.nn.moments(enc_output, axes=[0])
        tx = (enc_output - tf.cast(enc_output_mean, tf.float32)) / tf.cast(tf.sqrt(enc_output_var), enc_output.dtype)
    else:
        raise ValueError('{} is not a supported power constraint choice!'.format(pc_choice))
    return tx


def awgn(tx, config, noisr_var):
    """Implements the real AWGN channel.
        Args:
            tx: transmitted signals
            config: designated configurations
        Returns:
            y: noisy channel output signals
    """

    noise = tf.random_normal(tf.shape(tx), mean=config.noise_mean,
                             stddev=tf.sqrt(noisr_var), dtype=tx.dtype)
    y_out = tf.add(tx, noise, name='y_in')
    return y_out


class SOURCE:
    def __init__(self, config):
        self.config = config

    def generate(self):
        src_choice = self.config.src_dist
        if src_choice in ['uniform', 'unif']:
            return self._uniform()
        elif src_choice in ['normal', 'norm']:
            return self._normal()
        else:
            raise ValueError('Not support this source choice for now!')

    def _normal(self):
        if self.config.task.lower() in ['test']:
            bs = self.config.bs_test
        else:
            bs = self.config.bs
        dim = self.config.input_dim
        return np.random.normal(0., 1., size=(bs, dim))

    def _uniform(self):
        if self.config.task.lower() in ['test']:
            bs = self.config.bs_test
        else:
            bs = self.config.bs
        dim = self.config.input_dim
        return np.random.uniform(-1., 1., size=(bs, dim))
