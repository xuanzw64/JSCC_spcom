#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : train.py
author: Ziwei Xuan
email : xuan64@tamu.edu

Construct the network and train.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import os
import csv
import pandas as pd
from utils.model import *


def AJSCC(config):
    # construct graph
    dim = config.input_dim
    x_in = tf.placeholder(tf.float32, [None, dim], name='x_in')
    training = tf.placeholder_with_default(True, shape=(), name='training')
    noise_var = tf.placeholder(tf.float32, name='noise_var')

    # encoder
    encoder = Encoder(config)
    enc_output = encoder(x_in, training)
    # power constraint
    tx = power_constraint(enc_output, config)
    # AWGN channel
    y_out = awgn(tx, config, noise_var)
    # decoder
    decoder = Decoder(config)
    dec_output = decoder(y_out, training)
    # loss
    loss_mse = tf.reduce_mean(tf.pow(dec_output - x_in, 2))
    loss = loss_mse

    # learning rate schedule
    iter_per_epoch_train = config.iter
    epochs = config.epoch
    lr_base = tf.placeholder(tf.float32, name='lr')
    global_steps = tf.Variable(0, trainable=False)
    lr_decay_factor = config.staircase_decay_factor

    if config.lr_schedule == 'WRCosine':
        first_epoch_decay = config.first_epoch_decay
        t_mul = config.t_mul
        m_mul = config.m_mul
        schedule_global_steps = global_steps
        schedule = tf.train.cosine_decay_restarts(1., schedule_global_steps, first_epoch_decay * iter_per_epoch_train,
                                                  t_mul=t_mul, m_mul=m_mul)
        lr_opt = lr_base * schedule
        if config.weight_decay:
            epoch_recomp = tf.cast(global_steps, tf.float32) / (iter_per_epoch_train * first_epoch_decay)
            i_restart = tf.floor(tf.log(1. - epoch_recomp * (1. - t_mul)) / tf.log(t_mul * 1.))
            T_cur = first_epoch_decay * t_mul ** i_restart
            schedule_wd = schedule / (m_mul ** i_restart)
    elif config.lr_schedule == 'StaircaseDecay':
        stair_base = config.stair_base
        stair_boundary = stair_base * iter_per_epoch_train * epochs
        if type(lr_decay_factor) == np.float:
            stair_values = lr_decay_factor ** (np.arange(len(stair_base)) + 1.)
            stair_values = np.concatenate([[1.], stair_values])
        elif len(lr_decay_factor) == len(stair_base):
            stair_values = np.concatenate([[1.], lr_decay_factor])
        else:
            raise ValueError('Wrong lr_decay_factor setting!!')
        schedule = tf.train.piecewise_constant(global_steps, stair_boundary.astype(np.int).tolist(),
                                               stair_values.tolist())
        lr_opt = lr_base * schedule
        if config.weight_decay:
            T_cur = epochs
            schedule_wd = 1.
    else:
        lr_opt = lr_base
        if config.weight_decay:
            T_cur = epochs
            schedule_wd = 1.

    # optimizer schedule
    opt_choice = config.optimizer
    momentum = config.sgd_momentum
    nestrov = config.sgd_nestrov
    clip_norm = config.clip_norm

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        if config.weight_decay:
            w_norm = config.weight_decay_factor
            wd_norm_choice = config.weight_decay_norm
            wd = w_norm * tf.sqrt(0.01 / T_cur) * schedule_wd if wd_norm_choice else w_norm

        if config.weight_decay:
            opt = tf.contrib.opt.MomentumWOptimizer(learning_rate=lr_opt, weight_decay=wd, momentum=momentum,
                                                    use_nesterov=nestrov) if opt_choice == 'SGD' else tf.contrib.opt.AdamWOptimizer(
                learning_rate=lr_opt, weight_decay=wd)
        else:
            opt = tf.train.MomentumOptimizer(lr_opt, momentum,
                                             use_nesterov=nestrov) if opt_choice == 'SGD' else tf.train.AdamOptimizer(
                lr_opt)
        # gradient clip
        opt_grad, opt_var = zip(*opt.compute_gradients(loss))
        opt_grad, _ = tf.clip_by_global_norm(opt_grad, clip_norm)
        trainer_all = opt.apply_gradients(zip(opt_grad, opt_var), global_steps)

    # for saving the model
    saver = tf.train.Saver()
    opt_task = config.task
    lr_init = config.lr
    iter_per_epoch_val = config.iter_val
    iter_per_epoch_test = config.iter_test

    # set record
    AE = {'loss': [], 'loss_val': [], 'loss_test': [], 'lr': [], 'best': [np.inf]}

    # start the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if opt_task.lower() not in ['test']:
            train_noise_vars = 10. ** (-(config.snr + config.snr_train_offset) / 10.)
            test_noise_vars = 10. ** (-(config.snr + config.snr_test_offset) / 10.)
            # judge if or not to load / continue previous model
            if opt_task.lower() in ['continue']:
                CKPT_PATH = config.ckpt_path
                saver.restore(sess, CKPT_PATH)
                print('Continue training using saved model with CSNR {} dB...'.format(config.snr))
            else:
                print('Start new training with CSNR {} dB...'.format(config.snr))

            # set up the summary handle
            summary_writer = tf.summary.FileWriter(
                '{}/AJSCC_{}dB_k{}n{}/TB_summary'.format(config.save_path, config.snr, dim, config.output_dim),
                graph=tf.get_default_graph())
            source = SOURCE(config)

            dirs_rec = config.record_path + '/AJSCC_{}dB_k{}n{}'.format(config.snr, dim, config.output_dim)
            if not os.path.exists(dirs_rec):
                os.makedirs(dirs_rec)

            print('Begin Joint Training, {} iterations per epoch...'.format(iter_per_epoch_train))
            for epoch in range(int(epochs)):
                lr_schedule = lr_init

                # Train Stage
                AE_train = {'loss': []}

                # Joint Train:
                for it in range(int(iter_per_epoch_train)):
                    src_sample = source.generate()
                    _, l_joint, lr_opt_item = sess.run([trainer_all, loss, lr_opt],
                                                       feed_dict={x_in: src_sample, lr_base: lr_schedule,
                                                                  training: True, noise_var: train_noise_vars})
                    AE_train['loss'].append(l_joint)
                    AE['lr'].append(lr_opt_item)

                # Validation Stage
                AE_val = {'loss': [], 'csnr': []}
                for it in range(int(iter_per_epoch_val)):
                    src_sample = source.generate()
                    l_val = sess.run([loss_mse],
                                     feed_dict={x_in: src_sample, training: False, noise_var: test_noise_vars})
                    AE_val['loss'].append(l_val)

                # Record
                l_train_epoch = 10. * np.log10(np.array(AE_train['loss']).mean())
                l_val_epoch = 10. * np.log10(np.array(AE_val['loss']).mean())
                AE['loss'].append(l_train_epoch)
                AE['loss_val'].append(l_val_epoch)

                print("{:05d}/{:05d} epochs".format(epoch, epochs), "----",
                      " train loss = {:.5f} dB,".format(l_train_epoch), "||  Val loss = {:.5f} dB.".format(l_val_epoch))

                # save the model
                val_improve_check = False
                record_val_epoch = l_val_epoch
                if record_val_epoch < AE['best'][-1]:
                    saver.save(sess, '{}/AJSCC_{}dB_k{}n{}/model'.format(config.save_path, config.snr, dim, config.output_dim))
                    AE['best'].append(record_val_epoch)
                    print('best model saved...lr = {}'.format(lr_opt_item))
                    val_improve_check = True

                # Tensorboard Summary with self-created tag
                summary = tf.Summary()
                summary.value.add(tag="loss_train", simple_value=l_train_epoch)
                summary.value.add(tag="loss_val", simple_value=l_val_epoch)
                summary.value.add(tag="lr", simple_value=lr_opt_item)
                summary.value.add(tag="best_val", simple_value=AE['best'][-1])
                summary_writer.add_summary(summary, epoch)

                ''' recording during training'''
                if (epoch % 50 == 0 and epoch <= 100) or val_improve_check:
                    src_sample = source.generate()
                    tx_rec, l_rec = sess.run([tx, loss_mse],
                                             feed_dict={x_in: src_sample, training: False, noise_var: test_noise_vars})
                    bs = config.bs
                    if config.input_dim == 1:
                        src_unif = np.linspace(-3, 3, bs, endpoint=True).reshape((bs, 1))
                    elif config.input_dim == 2:
                        bs_root = int(np.sqrt(bs))
                        x_tmp, y_tmp = np.meshgrid(np.linspace(-3, 3, bs_root, endpoint=True),
                                                   np.linspace(-3, 3, bs_root, endpoint=True))
                        src_unif = np.concatenate([x_tmp.reshape((-1, 1)), y_tmp.reshape((-1, 1))], axis=-1)
                    else:
                        src_unif = np.random.uniform(-3., 3., size=(bs, config.input_dim))
                    tx_unif = sess.run([tx], feed_dict={x_in: src_unif, training: False})
                    a_set = ('source.npy', 'enc.npy', 'mse.npy', 'source_unif.npy', 'tx_unif.npy')
                    b_set = (src_sample, tx_rec, l_rec, src_unif, tx_unif)
                    for a, b in zip(a_set, b_set):
                        np.save(dirs_rec + '/' + a, b)
                    else:
                        best_rec = np.array([AE['best'][-1], epoch])
                    np.save(dirs_rec + '/best_record.npy', best_rec)
                    np.save(dirs_rec + '/loss_train.npy', np.array(AE['loss']))

            fp = '{}/result_record.csv'.format(dirs_rec)
            with open(fp, 'w', newline='') as fout:
                w = csv.writer(fout)
                for key, value in AE.items():
                    w.writerow([key, *value])

            summary_writer.close()

        elif opt_task == 'test':
            # Test Stage
            sweep_target = config.sweep_target
            test_path = config.test_path
            saver.restore(sess, test_path)
            print("Evaluate pre-trained model. Sweep target is {}. Model restored...".format(sweep_target.upper()))

            if sweep_target == 'snr':
                snr_range = config.snr_range
                AE_test = {'snr': snr_range, 'loss': []}
                source = SOURCE(config)
                for snr_item in snr_range:
                    AE_test_epoch = {'loss': [], 'csnr': []}
                    test_noise_vars = 10. ** (-(snr_item + config.snr_test_offset) / 10.)
                    for iter_test in range(int(iter_per_epoch_test)):
                        src_sample = source.generate()
                        l_val = sess.run([loss_mse],
                                         feed_dict={x_in: src_sample, training: False, noise_var: test_noise_vars})
                        AE_test_epoch['loss'].append(l_val)
                    l_test = 10. * np.log10(np.array(AE_test_epoch['loss']).mean())
                    AE_test['loss'].append(l_test)
            else:
                raise ValueError('Not support this sweep method now!')
            AE_test_df = pd.DataFrame(data=AE_test)
            AE_test_df.to_csv('{}/Test_result.csv'.format(config.record_path))
            print("Test ---- finished")

        else:
            raise ValueError('Wrong task value!')

    return



