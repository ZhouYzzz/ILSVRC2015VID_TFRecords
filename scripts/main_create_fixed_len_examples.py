"""
An example script that extract all the streams from ILSVRC2015VID, slice them into fixed-len streams,
then store them in TFRecords files as fixed_len_sequence_examples.
"""
import argparse
import os
import tqdm

from utils.stream import StreamSeparator
from dataset.annotations import AnnoMeta, AnnoObj, AnnoStream, parse_xml
from dataset.dataset import ILSVRC2015VID


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_dir', default=None, type=str, help='path to ILSVRC2015 root dir')
parser.add_argument('--output_dir', default=None, type=str, help='path to store generated tfrecords')
parser.add_argument('--phase', default='train', choices=['train', 'val', 'test', 'all'], type=str, help='choose phase')
parser.add_argument('--length', default=32, type=int, help='length of each sequence example')
parser.add_argument('--min_num_examples', default=2, type=int, help='minimum number of examples from single stream')
parser.add_argument('--max_num_examples', default=16, type=int, help='maximum number of examples from single stream')


FLAGS = parser.parse_args(['--dataset_dir', '/home/zhouyz/ILSVRC2015'])


def parse_annotation_folder(folder):
  def split_stream(s: AnnoStream):
    return s.split(n=max(min(s.length // FLAGS.length + 1, FLAGS.max_num_examples), FLAGS.min_num_examples),
                   l=FLAGS.length)
  if not os.path.exists(folder):
    raise NotADirectoryError(folder)
  num_xml_files = len([f for f in os.listdir(folder) if f.endswith('.xml')])
  stream_separator = StreamSeparator(dtype=AnnoObj)
  for i in range(num_xml_files):
    meta, objs = parse_xml(os.path.join(folder, '{:06d}.xml'.format(i)))
    stream_separator.update({o.trackid: o for o in objs})
  streams = stream_separator.close()
  anno_streams = [AnnoStream.from_meta_and_objs(meta=meta, objs=s) for s in streams]
  splitted_streams = list()
  [splitted_streams.extend(split_stream(s)) for s in anno_streams]
  return splitted_streams


def main():
  dataset = ILSVRC2015VID(dataset_dir=FLAGS.dataset_dir)
  sids = dataset.get_sequence_ids(phase=FLAGS.phase)
  for sid in tqdm.tqdm(sids):
    streams = parse_annotation_folder(os.path.join(dataset.annotation_dir(FLAGS.phase, sid)))


if __name__ == '__main__':
  main()
