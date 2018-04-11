"""
```content of file ILSVRC2015_val_00030000/000000.xml
<annotation>
  <folder>ILSVRC2015_val_00030000</folder>
  <filename>000000</filename>
  <source>
    <database>ILSVRC_2015</database>
  </source>
  <size>
    <width>1280</width>
    <height>720</height>
  </size>
  <object>
    <trackid>0</trackid>
    <name>n02510455</name>
    <bndbox>
      <xmax>855</xmax>
      <xmin>449</xmin>
      <ymax>682</ymax>
      <ymin>304</ymin>
    </bndbox>
    <occluded>1</occluded>
    <generated>0</generated>
  </object>
  <object>
    <trackid>1</trackid>
    <name>n02510455</name>
    <bndbox>
      <xmax>931</xmax>
      <xmin>172</xmin>
      <ymax>590</ymax>
      <ymin>307</ymin>
    </bndbox>
    <occluded>1</occluded>
    <generated>0</generated>
  </object>
</annotation>
```
"""
from collections import namedtuple
from xml.etree.ElementTree import parse, Element
from utils.slice import fixed_len_slice_duplicated_pad
from utils.stream import StreamSeparator
from typing import List
import tensorflow as tf
from tfhelper.features import *


class AnnoMeta(namedtuple('Annometa', ['folder', 'filename', 'size'])):
  """AnnoMeta
  The meta information (or header) contained in every annotation file (*.xml), which
  is consistent along video frames.

  Fields:
    folder (bytes): a relative path directing to the folder of the sequence
    filename (bytes): a 6-len digits representing the filename/frame no., not used
    size (tuple(int64, int64)): a tuple representing the image size, (width, height)
  """
  @classmethod
  def from_xml(cls, elem: Element):
    """Create a new AnnoMeta from an xml Element, which is the root Element parsed from annotation file"""
    if not elem.tag == 'annotation':
      raise ValueError('`elem` is not tagged with `annotation`, got {}'.format(elem.tag))
    return super(AnnoMeta, cls).__new__(cls,
                                        folder=bytes(elem[0].text, 'utf-8'),
                                        filename=bytes(elem[1].text, 'utf-8'),
                                        size=(int(elem[3][0].text), int(elem[3][1].text)))


class AnnoObj(namedtuple('AnnoObj', ['trackid', 'filename', 'name', 'bndbox', 'occluded', 'generated'])):
  """AnnoObj
  The object information. Each frame may contain multiple objects, each identified by the key `trackid`.
  A stream contains a continuous sequence of the same object appearing in different frames.

  Fields:
    trackid (bytes): the key field to identify each object
    filename (bytes): a 6-len digits representing the filename/frame no., shared with AnnoMeta
    name (bytes): the identifier of the object in the whole dataset
    bndbox ([int64, int64, int64. int64]): the bounding box that fits the object, [xmax, xmin, ymax, ymin]
    occluded (int64): bool, not used
    generated (int64): bool, not used
  """
  @classmethod
  def from_xml(cls, elem: Element, filename: bytes):
    """Create a new AnnoObj from an xml Element tagged `object`, which is extracted from the parsed annotation file"""
    if not elem.tag == 'object':
      raise ValueError('`elem` is not tagged with `object`, got {}'.format(elem.tag))
    return super(AnnoObj, cls).__new__(cls,
                                       trackid=bytes(elem[0].text, 'utf-8'),
                                       filename=filename,
                                       name=bytes(elem[1].text, 'utf-8'),
                                       bndbox=[int(e.text) for e in elem[2][0:4]],
                                       occluded=int(elem[3].text),
                                       generated=int(elem[4].text))


def parse_xml(xml_filename):
  """parse_xml
  Parse ILSVRC2015VID Annotation file

  :param xml_filename: ILSVRC2015VID annotation filename
  :return: (AnnoMeta, List[AnnoObj]) tuple
  """
  def _extract_obj_elems(root: Element):
    return root[4:]
  tree = parse(xml_filename)
  root = tree.getroot()
  meta = AnnoMeta.from_xml(root)
  objs = list(map(lambda e: AnnoObj.from_xml(e, filename=meta.filename), _extract_obj_elems(root)))
  return meta, objs


class AnnoStream(namedtuple('AnnoStream', ['meta', 'length', 'filenames', 'bndboxes'])):
  """AnnoStream
  The basic class representing a video snippet of a single object.
  Can be used in Object Tracking tasks.

  Fields:
    meta (AnnoMeta): the meta information of the stream
    length (int64): the length of the stream
    filenames (List[bytes]): a list of filenames that contain the object
    bndboxes (List[4 * int64]): a list of 4 int64 representing the object bounding boxes
  """
  @classmethod
  def from_meta_and_objs(cls, meta: AnnoMeta, objs: List[AnnoObj]):
    return super(AnnoStream, cls).__new__(cls,
                                          meta=meta,
                                          length=len(objs),
                                          filenames=[o.filename for o in objs],
                                          bndboxes=[o.bndbox for o in objs])

  def substream(self, s: int, l: int):
    """Extract a `l`-len substream starts at `s`"""
    filenames = fixed_len_slice_duplicated_pad(self.filenames, s, l)
    bndboxes = fixed_len_slice_duplicated_pad(self.bndboxes, s, l)
    return super(AnnoStream, self).__new__(self.__class__,
                                           meta=self.meta,
                                           length=l,
                                           filenames=filenames,
                                           bndboxes=bndboxes)

  def split(self, n: int, l: int):
    """Split into `n` substreams len `l`"""
    itv = max(self.length // n, 0)  # interval
    ss = map(lambda x: itv * x, range(0, n))  # starts
    return [self.substream(s, l) for s in ss]

  def as_sequence_example(self):
    """Convert to tf.train.SequenceExample"""
    return tf.train.SequenceExample(
      context=tf.train.Features(feature={
        'folder': feature_bytes(self.meta.folder),
        'size': feature_int64(self.meta.size),
        'length': feature_int64(self.length)
      }),
      feature_lists=tf.train.FeatureLists(feature_list={
        'filenames': feature_list_bytes(self.filenames),
        'bndboxes': feature_list_int64(self.bndboxes)
      })
    )


def parse_annotation_folder_to_streams(folder: str) -> List[AnnoStream]:
  """parse_annotation_folder_to_streams
  Read the whole annotation folder, split the sequence into streams, 
  and return all the streams in the format of AnnoStream

  return:
    List of AnnoStream
  """
  if not os.path.exists(folder):
    raise NotADirectoryError(folder)
  num_xml_files = len([f for f in os.listdir(folder) if f.endswith('.xml')])
  stream_separator = StreamSeparator(dtype=AnnoObj)
  for i in range(num_xml_files):
    meta, objs = parse_xml(os.path.join(folder, '{:06d}.xml'.format(i)))
    stream_separator.update({o.trackid: o for o in objs})
  streams = stream_separator.close()
  anno_streams = [AnnoStream.from_meta_and_objs(meta=meta, objs=s) for s in streams]
  return anno_streams


class AnnoScene(namedtuple('AnnoScene', ['meta', 'num_object', 'bndboxes', 'classes'])):
  """AnnoScene
  The basic class representing a single frame containing multiple objects.
  Cam be used in Object Detection tasks.
  """
  pass
