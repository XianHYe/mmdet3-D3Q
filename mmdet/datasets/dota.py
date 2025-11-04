from .coco import CocoDataset

import itertools
import logging
from collections import OrderedDict

import numpy as np

from aitodpycocotools.cocoeval import COCOeval


from mmdet.registry import DATASETS
from .coco import CocoDataset



@DATASETS.register_module()
class DOTA2Dataset(CocoDataset):
    METAINFO = {
        'classes':
                ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
               'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad')
    }


