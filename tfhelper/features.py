import tensorflow as tf
from typing import Sequence, Union, List


def feature_int64(value: Union[int, Sequence[int]]) -> tf.train.Feature:
  if isinstance(value, int):
    value=[value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def feature_float(value: Union[float, Sequence[float]]) -> tf.train.Feature:
  if isinstance(value, float):
    value=[value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def feature_bytes(value: Union[str, bytes, Sequence[bytes]]) -> tf.train.Feature:
  if isinstance(value, str):
    value = bytes(value, 'utf-8')
  if isinstance(value, bytes):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def feature_list_int64(values: List):
  return tf.train.FeatureList(feature=[feature_int64(v) for v in values])


def feature_list_float(values: List):
  return tf.train.FeatureList(feature=[feature_float(v) for v in values])


def feature_list_bytes(vlaues: List):
  return tf.train.FeatureList(feature=[feature_bytes(v) for v in vlaues])
