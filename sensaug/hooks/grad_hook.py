"""
hook for storing gradient data for augmentations preformed during training
"""

import os
import json
import warnings
from pprint import pprint
from copy import deepcopy
from typing import Optional, Sequence
from mmengine.visualization import Visualizer

import mmcv
from mmengine.fileio import get
from mmengine.runner import Runner
from mmseg.registry import HOOKS
from mmengine.hooks import Hook
from mmseg.structures import SegDataSample
from mmseg.engine.hooks.visualization_hook import SegVisualizationHook

# Local imports
from sensaug.dataset.augmentations import *
from sensaug.sensitivity_analysis import *
from sensaug.runner_utils import *

@Hooks.register_module()
class CollectGradientHook(Hook): 
    
    def __init__(self) -> None:
        pass

    # after training on an iter runs the step to collect gradients
    def after_train_iter(
            self,
            runner: Runner,
            batch_idx, int,
            dtata_batch=None
    ) -> None: 
        pass
 
