
from .coco import CocoDataset

import itertools
import logging
from collections import OrderedDict

import numpy as np

from aitodpycocotools.cocoeval import COCOeval


from mmdet.registry import DATASETS
from .coco import CocoDataset



@DATASETS.register_module()
class VISDRONEDataset(CocoDataset):
    METAINFO = {
        'classes':
                ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
    }


