from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmdet.datasets.builder import DATASETS, PIPELINES
from multiprocessing import freeze_support
from mmcv import Config
import torch
from mmdet.apis import inference_detector, init_detector, train_detector, show_result_pyplot
import cv2
from mmdet.core.evaluation import eval_map
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel



if __name__=='__main__':
    freeze_support()
    config_file = './custum_config_save.py'
    cfg = Config.fromfile(config_file)
    checkpoint_file = './custum_pretrained/latest.pth'
    # 검증 데이터셋을 빌드합니다.
    val_dataset = build_dataset(cfg.data.val)

    # 검증 데이터 로더를 빌드합니다.
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,  # 배치 크기 설정
        workers_per_gpu=1,
        dist=False,
        shuffle=False)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # 검증 데이터셋을 사용하여 모델을 평가합니다.

    dataset = val_loader.dataset
    model.cfg = cfg
    det_results = []

    for i, data in enumerate(dataset):
        with torch.no_grad():
            img = cv2.imread(data['img_metas'][0].data['filename'])
            result = inference_detector(model, img)
            det_results.append(result)
    metrics = val_dataset.evaluate(det_results)
    print(metrics)