from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cell_output_state_wrapper(cell_cls, name, state_is_tuple=False):
  attr_dict = {}
  def __call__(self, inputs, state, scope=None):
    output, state = cell_cls.__call__(self, inputs, state, scope)
    if isinstance(state, tuple):
      output = tf.concat([s for s in state], axis=1)
      return output, state
    else:
      return state, state

  @property
  def output_size(self):
    if state_is_tuple:
      return sum(self.state_size)
    else:
      return self.state_size

  attr_dict = {'output_size':output_size, '__call__': __call__}
  return type(name, (cell_cls,), attr_dict)
