from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import time

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test

from sgrnn import reader

class PDBTest(test.TestCase):
  def test_pdb_state_saver(self):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(
      dir_path, '..', '..', '..', 'data',
      'simple-examples', 'data')
    train_data, valid_data, test_data, vocabulary = reader.ptb_raw_data(data_path)
    init_states = {'state1': np.zeros(100)}
    batch = reader.pdb_state_saver(raw_data=test_data, batch_size=8, num_steps=64, init_states=init_states,
      num_unroll=8, num_threads=3, capacity=1000, allow_small_batch=False, name=None)

class BarrierTest(test.TestCase):
  def testInsertMany(self):
    with self.test_session():
      ba = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), name="B")
      size_t = ba.ready_size()
      self.assertEqual([], size_t.get_shape())
      keys = [b"a", b"b", b"c"]
      insert_0_op = ba.insert_many(0, keys, [10.0, 20.0, 30.0])
      insert_1_op = ba.insert_many(1, keys, [100.0, 200.0, 300.0])

      self.assertEquals(size_t.eval(), [0])
      insert_0_op.run()
      self.assertEquals(size_t.eval(), [0])
      insert_1_op.run()
      self.assertEquals(size_t.eval(), [3])
  
  def testTakeMany(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), name="B")
      size_t = b.ready_size()
      keys = [b"a", b"b", b"c"]
      values_0 = [10.0, 20.0, 30.0]
      values_1 = [100.0, 200.0, 300.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys, values_1)
      take_t = b.take_many(3)

      insert_0_op.run()
      insert_1_op.run()
      self.assertEquals(size_t.eval(), [3])

      vals = sess.run(take_t)

      # indices_val, keys_val, values_0_val, values_1_val = sess.run(
      #     [take_t[0], take_t[1], take_t[2][0], take_t[2][1]])

    self.assertAllEqual(indices_val, [-2**63] * 3)
    for k, v0, v1 in zip(keys, values_0, values_1):
      idx = keys_val.tolist().index(k)
      self.assertEqual(values_0_val[idx], v0)
      self.assertEqual(values_1_val[idx], v1)

    assert False


def test_seq_generator():
  max_len = 11
  num_feat = 3
  seq_len = tf.cast(tf.ceil(tf.random_uniform([2]) * max_len), dtype=tf.int32)
  ds = tf.data.Dataset.from_tensor_slices(tf.random_uniform(tf.concat([seq_len, [num_feat]], 0)))

  print(ds.output_types)
  print(ds.output_shapes)
  print(seq_len)


class TestSeq(tf.test.TestCase):
  def setUp(self):
    pass

  def test_seq(self):
    n_data = 23
    max_len = 11
    num_feat = 3
    seq_len = tf.cast(tf.ceil(tf.random_uniform([1]) * max_len), dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices(tf.random_uniform(tf.concat([[n_data], seq_len, [num_feat]], 0)))
    ds2 = ds.skip(1)
    ds2_final = ds.take(1)
    ds2 = ds2.concatenate(ds2_final)
    seg_type = tf.data.Dataset.from_tensors(tf.zeros([1], tf.int32))
    seg_type_1 = tf.data.Dataset.from_tensors(tf.ones([1], dtype=tf.int32)).repeat(n_data - 2)
    seg_type_2 = tf.data.Dataset.from_tensors(tf.ones([1], dtype=tf.int32) * 2)
    seg_type = seg_type.concatenate(seg_type_1).concatenate(seg_type_2)

    ds = tf.data.Dataset.zip((ds, ds2, seg_type))

    print(ds.output_types)
    print(ds.output_shapes)
    print(seq_len)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      val = sess.run(seq_len)
      print(val)


