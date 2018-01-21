
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
from tensorflow.contrib.training import batch_sequences_with_states

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y


def _circular_shift(x, step_size, axis):
  with tf.name_scope("circular_shift"):
    size = tf.shape(x)[axis]
    x0, x1 = tf.split(x, [step_size, size - step_size], axis=axis)
    x = tf.concat([x1, x0], axis=axis)
  return x


def pdb_state_saver(raw_data, batch_size, num_steps, init_states,
  num_unroll, num_threads=3, capacity=1000, allow_small_batch=False, name=None):
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    data_len = tf.size(raw_data)
    n_seq = (data_len - 1) // num_steps

    # need to make sure the num_step is multiple of num_unroll
    raw_data_x = tf.reshape(raw_data[0 : n_seq * num_steps],
                      [n_seq, num_steps])
    next_raw_data_x = _circular_shift(raw_data_x, num_unroll, axis=1)
    raw_data_y = tf.reshape(raw_data[1 : (n_seq * num_steps + 1)],
                      [n_seq, num_steps])
    next_raw_data_y = _circular_shift(raw_data_y, num_unroll, axis=1)
    keys = tf.range(n_seq)
    keys = tf.cast(keys, dtype=tf.string)
    seq_len = tf.tile([num_steps], [n_seq])
    data = tf.data.Dataset.from_tensor_slices(
      (keys, raw_data_x, next_raw_data_x, raw_data_y, next_raw_data_y, seq_len))
    iterator = data.make_initializable_iterator()
    next_key, next_x, next_next_x, next_y, next_next_y, next_len = iterator.get_next()
    seq_dict = {'x':next_x, 'next_x':next_next_x, 'y':next_y, 'next_y':next_next_y}
    batch = batch_sequences_with_states(
      input_key=next_key,
      input_sequences=seq_dict,
      input_context={},
      input_length=next_len,
      initial_states=init_states,
      num_unroll=num_unroll,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=capacity,
      allow_small_batch=allow_small_batch,
      pad=True)
  return batch


def seq_generator():
  max_len = 11
  num_feat = 3
  seq_len = tf.cast(tf.ceil(tf.random.uniform() * max_len), dtype=tf.int32)
  ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform([seq_len, num_feat]))
