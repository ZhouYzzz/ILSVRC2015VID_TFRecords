import tensorflow as tf
import glob
import os


def parse_image(dataset_dir_t, folder_t, filename_t):
  image_filename = tf.string_join([dataset_dir_t, folder_t, tf.string_join([filename_t[0], '.JPEG'])], separator='/')
  image_raw = tf.read_file(image_filename)
  image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
  image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  image_resized = tf.image.resize_images(image_decoded, [360, 480])
  return image_resized


def parse_bndbox(bndboxes_t, size_t):
  bndboxes_t = tf.cast(bndboxes_t, dtype=tf.float32)
  size_t = tf.cast(size_t, dtype=tf.float32)
  bndboxes_t = tf.div(bndboxes_t, [size_t[0], size_t[0], size_t[1], size_t[1]])
  return bndboxes_t


def preprocess(context, sequence, dataset_dir):
  sequence['images'] = tf.map_fn(
    lambda filename: parse_image(dataset_dir, context['folder'], filename),
    sequence['filenames'],
    dtype=tf.float32
  )
  sequence['bndboxes'] = tf.map_fn(
    lambda bndboxes: parse_bndbox(bndboxes, context['size']),
    sequence['bndboxes'],
    dtype=tf.float32
  )
  return context, sequence


def fixed_len_input_fn(records_dir,
                       batch_size,
                       num_epoches,
                       dataset_dir):
  tfrecord_files = glob.glob(os.path.join(records_dir, '*.tfrecords'))
  if len(tfrecord_files) == 0:
    raise FileNotFoundError('No existing tfrecords. Run generating script first.')
  dataset = tf.data.TFRecordDataset(tfrecord_files)  # type: tf.data.Dataset
  dataset = dataset.map(
    lambda s: tf.parse_single_sequence_example(s,
                                               context_features={
                                                 'folder': tf.FixedLenFeature([], tf.string),
                                                 'size': tf.FixedLenFeature([2], tf.int64),
                                                 'length': tf.FixedLenFeature([], tf.int64)
                                               },
                                               sequence_features={
                                                 'filenames': tf.FixedLenSequenceFeature([1], tf.string),
                                                 'bndboxes': tf.FixedLenSequenceFeature([4], tf.int64)
                                               }))
  dataset = dataset.map(lambda c, s: preprocess(context=c, sequence=s, dataset_dir=dataset_dir))
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.repeat(count=num_epoches)
  iterator = dataset.make_one_shot_iterator()
  context, sequence = iterator.get_next()
  return context, sequence
