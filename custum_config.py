from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv


config_file = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
checkpoint_file = './mmdetection/checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

cfg = Config.fromfile(config_file)
cfg.dataset_type = 'PolypDataset'
cfg.data_root = './polypdata/'
cfg.device = 'cuda'
cfg.runner.max_epochs = 20
# 설정 파일 내의 일부 내용
log_level = 'INFO'  # 로그 레벨 설정 (DEBUG, INFO, WARNING 등)
load_from = None    # 이전에 학습된 모델을 불러오는 설정
resume_from = None  # 중단된 학습을 재개하는 설정

cfg.data.train.dataset.type = 'PolypDataset'
cfg.data.train.dataset.data_root = './polypdata/'
cfg.data.train.dataset.ann_file = 'train.txt'
cfg.data.train.dataset.img_prefix = 'images/'

cfg.data.val.type = 'PolypDataset'
cfg.data.val.data_root = './polypdata/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'images/'

cfg.data.test.type = 'PolypDataset'
cfg.data.test.data_root = './polypdata/'
cfg.data.test.ann_file = 'train.txt'
cfg.data.test.img_prefix = 'images/'

cfg.model.roi_head.bbox_head.num_classes = 1

cfg.load_from = checkpoint_file

cfg.work_dir = './custum_pretrained'

cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10
cfg.lr_config.policy = 'step'
cfg.evaluation.metric = 'mAP'

cfg.evaluation.interval = 10
cfg.checkpoint_config.interval = 10
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

cfg.dump('./custum_config_save.py')