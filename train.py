from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from multiprocessing import freeze_support
import os.path as osp
import cv2
import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.apis import set_random_seed

cfg = Config.fromfile('./custum_config_save.py')

if __name__=='__main__':
    freeze_support()
    datasets = [build_dataset(cfg.data.train.dataset)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES
    print(datasets[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    train_detector(model, datasets, cfg, distributed=False, validate=True)
    img = cv2.imread('./polypdata/images/cju0qkwl35piu0993l0dewei2.jpg')
    model.cfg = cfg

    result = inference_detector(model, img)
    show_result_pyplot(model, img, result)

