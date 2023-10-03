from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from multiprocessing import freeze_support
import os.path as osp
import cv2
import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.apis import single_gpu_test

if __name__=='__main__':
    freeze_support()
    config_file = './custum_config_save.py'
    cfg = Config.fromfile(config_file)
    checkpoint_file = './custum_pretrained/epoch_1.pth'

    datasets = [build_dataset(cfg.data.train.dataset)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    img = cv2.imread('./polypdata/images/cju0qkwl35piu0993l0dewei2.jpg')
    model.cfg = cfg

    result = inference_detector(model, img, )
    show_result_pyplot(model, img, result)

