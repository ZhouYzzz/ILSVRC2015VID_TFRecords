ILSVRC2015VID_TFRecords
===

This repository is used to generate sequencial TFRecords files from raw ILSVRC2015VID datasets. The generated record files can be used to train networks used for Visual Tracking, Object Detection Tasks, e.t.c.

### Requirements

* python3.5 or later
* tensorflow-gpu == 1.4.1 (or tensorflow == 1.4.1)
* tqdm
* ILSVRC2015 VID dataset

run `pip install -r requirements.txt` to install required packages.

go to [ILSVRC2015](image-net.org/challenges/LSVRC/2015/) for more information and download the dataset.

### Create fixed length TFSequenceExample

run `scripts/main_create_fixed_len_examples.py`

use `tfhelper.input_fn.fixed_len_input_fn` for `tf.Estimator` API
