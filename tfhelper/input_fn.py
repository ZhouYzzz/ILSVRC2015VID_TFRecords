import tensorflow as tf
import glob
import os


def parse_image(dataset_dir_t, folder_t, filename_t, size=(360, 480)):
  """Convert filename str to RGB image data

  params:
    dataset_dir_t: Tensor(str) of dataset_dir, e.g. /path/to/ILSVRC2015/Data/VID/train
    folder_t: Tensor(str) of sequence folder
    filename_t: Tensor(str) '%06d' representing the frame number
    size: 2D tuple representing resized (height, width)

  return:
    Resized RGB images
  """
  image_filename = tf.string_join([dataset_dir_t, folder_t, tf.string_join([filename_t[0], '.JPEG'])], separator='/')
  image_raw = tf.read_file(image_filename)
  image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
  image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  image_resized = tf.image.resize_images(image_decoded, size)
  return image_resized


def parse_bndbox(bndboxes_t, size_t):
  """Convert absolute bndbox locations to normalized ones

  params:
    bndboxes_t: Tensor of bndboxes, (xmax, xmin, ymax, ymin)
    size_t: Tensor of image size, (x(w), y(h))
  returns:
    Tensor of normalized bndboxes
  """
  bndboxes_t = tf.cast(bndboxes_t, dtype=tf.float32)
  size_t = tf.cast(size_t, dtype=tf.float32)
  bndboxes_t = tf.div(bndboxes_t, [size_t[0], size_t[0], size_t[1], size_t[1]])
  return bndboxes_t


def preprocess(context, sequence, dataset_dir, size=(360, 480)):
  """Preprocess raw TF.SequenceExample.
  1. Read image data from disk
  2. Normalize bndboxes

  params:
    (context, sequence): the parsed TF.SequenceExample components
    dataset_dir: path to ILSVRC2015 data dir /*/*/Data/VID/train
    size: the desired size to resize raw images

  returns:
    Processed TF.SequenceExample components (context, sequence)
  """
  sequence['images'] = tf.map_fn(
    lambda filename: parse_image(dataset_dir, context['folder'], filename, size=size),
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
                       dataset_dir,
                       batch_size=8,
                       num_epoches=None,
                       image_resize_size=(360, 480)):
  """Fixed Len Input Function
  Used to provide train/val examples for TF.estimator API.

  params:
    records_dir: path to tfrecords files
    dataset_dir: path to ILSVRC2015 image files

  returns:
    (context, sequence) Tensors
  """
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
  dataset = dataset.map(
    lambda c, s: preprocess(context=c, sequence=s, dataset_dir=dataset_dir, size=image_resize_size))
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.repeat(count=num_epoches)
  iterator = dataset.make_one_shot_iterator()
  context, sequence = iterator.get_next()
  return context, sequence
