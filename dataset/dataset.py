import os
import csv


_VID = 'VID'


class _Path:
  Annotations = 'Annotations'
  ImageSets = 'ImageSets'


class _Pattern:
  pass


class ILSVRC2015VID(object):
  def __init__(self, dataset_dir):
    if not os.path.exists(dataset_dir):
      raise NotADirectoryError(dataset_dir)
    self._dataset_dir = dataset_dir

  @property
  def dataset_dir(self):
    return self._dataset_dir

  def annotation_dir(self, *args):
    return os.path.join(self.dataset_dir, _Path.Annotations, _VID, *args)

  def imageset_dir(self, *args):
    return os.path.join(self.dataset_dir, _Path.ImageSets, _VID, *args)

  def get_imageset_files(self, phase):
    if phase == 'train':
      return [self.imageset_dir('train_{}.txt'.format(i + 1)) for i in range(30)]
    elif phase == 'val':
      raise NotImplementedError
      # return [self.imageset_dir('val.txt')]
    elif phase == 'test':
      raise NotImplementedError
      # return [self.imageset_dir('test.txt')]
    else:
      raise ValueError('phase unrecgonized {}'.format(phase))

  def get_sequence_ids(self, phase):
    def _get_sequence_ids_from_file(filename):
      with open(filename, 'r') as csvfile:
        return [sid for sid, _ in csv.reader(csvfile, delimiter=' ')]
    sids = list()
    imagesets_files = self.get_imageset_files(phase=phase)
    for file in imagesets_files:
      sids.extend(_get_sequence_ids_from_file(file))
    return sids


if __name__ == '__main__':
  dataset = ILSVRC2015VID(dataset_dir='/home/zhouyz/ILSVRC2015/')
  print(len(dataset.get_sequence_ids('train')))
