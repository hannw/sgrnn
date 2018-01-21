from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.util import nest

from sgrnn import main
from sgrnn import reader


class BuildGraphTest(tf.test.TestCase):

  # def setUp(self):
  #   self.xs = tf.placeholder(
  #     tf.float32, shape=[batch_size, max_time, n_feats]) # [batch_size, max_time, n_feats]
  #   self.seq_len = tf.placeholder(tf.int32, shape=[batch_size])
  #   self.init_state = tf.placeholder(tf.float32, shape=[batch_size, n_feats])

  def test_main(self):
    config = main.TestConfig()
    model = main.PTBModel(config=config, is_training=True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(
      dir_path, '..', '..', '..', 'data',
      'simple-examples', 'data')
    raw_data = reader.ptb_raw_data(data_path)
    train_data, valid_data, test_data, _ = raw_data

    # init_states = {k:v for k, v in zip(model.state_name[0], model.zero_state[0])}
    # assert False
    # init_states = {model.state_name: model.zero_state}
    init_states = {k:v for k, v in zip(
      nest.flatten(model.state_name), nest.flatten(model.zero_state))}
    inputs = main.PTBInput(config=config, data=test_data, init_states=init_states)

    model.build_graph(inputs)
