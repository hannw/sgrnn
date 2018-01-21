
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from sgrnn import rnn_cell


class TestReturnState(tf.test.TestCase):
  def test_cell_output_state_wrapper(self):
    with self.test_session() as sess:
      batch_size = 8
      hidden_size = 7
      WrappedLSTM = rnn_cell.cell_output_state_wrapper(
        cell_cls=tf.contrib.rnn.BasicLSTMCell,
        name='WrappedLSTM', state_is_tuple=True)
      cell = WrappedLSTM(
          num_units=hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=False)

      inputs = tf.convert_to_tensor(
        np.random.rand(batch_size, 3), dtype=tf.float32)
      states = tuple([tf.zeros([batch_size, hidden_size], tf.float32)] * 2)
      output, next_states = cell(inputs, states)

      tf.global_variables_initializer().run()
      out_val, ns_val = sess.run([output, next_states])

      assert False