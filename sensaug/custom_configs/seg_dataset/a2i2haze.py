from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class A2IHazeDataset(BaseSegDataset):
    METAINFO = dict(classes=("xxx", "xxx", ...), palette=[[x, x, x], [x, x, x], ...])

    def __init__(self, arg1, arg2):
        pass
