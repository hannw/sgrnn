from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from sgrnn import main
from sgrnn import reader


class BuildGraphTest(tf.test.TestCase):

  def test_PTBInput(self):
    config = main.TestConfig()
    model = main.PTBModel(config=config, is_training=True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(
      dir_path, '..', '..', '..', 'data',
      'simple-examples', 'data')
    raw_data = reader.ptb_raw_data(data_path)
    train_data, valid_data, test_data, _ = raw_data
    inputs = main.PTBInput(config=config, data=test_data, init_states=model.init_state_dict)
    model.build_graph(inputs)
