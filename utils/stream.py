class StreamSeparator(object):
  """A helper util to split each ILSVRC2015 sequence into single-objected streams"""
  def __init__(self, dtype):
    self._dtype = dtype
    self._active_stream = dict()
    self._inactive_stream = list()

  def update(self, identified_dict):
    """identified_dict: dict of object {identity: dtype}

    Example: for a dict representing an object `obj`, which is identified with obj['id'],
             StreamSeperator.update({obj['id']: obj for obj in object_list})
    """
    present_ids = identified_dict.keys()
    active_ids = self._active_stream.keys()
    for i in list(present_ids - active_ids):
      self._active_stream[i] = list()
    for i in list(present_ids):
      if isinstance(identified_dict[i], self._dtype):
        self._active_stream[i].append(identified_dict[i])
      else:
        raise TypeError('Input dict values should be of type {}, get {}'.format(self._dtype, type(identified_dict[i])))
    for i in list(active_ids - present_ids):
      self._inactive_stream.append(self._active_stream.pop(i))

  def close(self):
    """close: Stop updating and deactivate all active streams, return all the separated streams"""
    active_ids = self._active_stream.keys()
    for i in list(active_ids):
      self._inactive_stream.append(self._active_stream.pop(i))
    return self._inactive_stream
