from DataLoader.dataset.sources import *
from setting.read_setting import config
class DataCache():
    def __init__(self) -> None:
        self.data_dir = config['coco_data_dir']
        self.trainval_source = COCOSource(data_dir=self.data_dir,
                                    mode='trainval',
                                    cache_type=config['cache_type'],
                                    cache=config['cache'])
        self.test_source=COCOSource(data_dir=self.data_dir,
                                    mode='test',
                                    cache_type=config['cache_type'],
                                    cache=config['cache'])
        self.mask_source=MaskSource(ids=self.trainval_source.send_ids(),
                                    data_dir=self.data_dir,
                                    cache_type=config['cache_type'],
                                    cache=config['cache'])