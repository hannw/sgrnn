from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from sgrnn import reader

flags = tf.flags

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")


batch_size = 7
max_time = 11
n_feats = 3
n_hidden = 23
output_size = 5

xs = tf.placeholder(
  tf.float32, shape=[batch_size, max_time, n_feats]) # [batch_size, max_time, n_feats]
seq_len = tf.placeholder(tf.int32, shape=[batch_size])
init_state = tf.placeholder(tf.float32, shape=[batch_size, n_feats])
targets = tf.placeholder(tf.int32, shape=[batch_size, max_time, output_size])


dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(
  dir_path, '..', '..', 'data',
  'simple-examples', 'data')
train_data, valid_data, test_data, vocabulary = reader.ptb_raw_data(data_path)
init_states = {'state1': np.zeros(100)}
batch = reader.pdb_state_saver(raw_data=train_data, batch_size=8, num_steps=100, init_states=init_states,
  num_unroll=10, num_threads=3, capacity=1000, allow_small_batch=False, name=None)
input_data = batch.sequences['x']
next_input_data = batch.sequences['next_x']
targets = batch.sequences['y']
next_targets = batch.sequnces['next_y']



cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size + n_hidden + n_hidden)
# outputs, final_state = tf.nn.dynamic_rnn(
#   cell=cell, inputs=xs, sequence_length=seq_len,
#   initial_state=init_state, dtype=tf.float32)


with tf.variable_scope('main_rnn') as scope:
  outputs, final_state = tf.nn.static_state_saving_rnn(
      cell=cell,
      inputs,
      state_saver,
      state_name,
      sequence_length=None
  )

with tf.variable_scope(scope, reuse=True):
  next_outputs, next_final_state = tf.nn.static_state_saving_rnn(
      cell,
      inputs,
      state_saver,
      state_name,
      sequence_length=None
  )

ys, sgs, aux_sgs = tf.split(outputs, [output_size, n_hidden, n_hidden], axis=2)
sg = tf.slice(sgs, begin=[0, 0, 0], size=[-1, 1, -1])
sg = tf.squeeze(sg, axis=1)
aux_sg = tf.slice(aux_sgs, begin=[0, 0, 0], size=[-1, 1, -1])
aux_sg = tf.squeeze(aux_sg, axis=1)


segment_loss = tf.contrib.seq2seq.sequence_loss(
  ys, targets,
  tf.sequence_mask(seq_len, max_time),
  average_across_timesteps=True,
  average_across_batch=True)







