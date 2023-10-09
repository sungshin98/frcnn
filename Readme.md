# custumdataset을 활용한 mmdet

파이썬 파일에 학습 돌리기전

if __name__=='__main__':
    freeze_support()
    
를 입력 후 train모델을 설정하고 학습

# PolypDataset.py
./mmdetection/mmdet/datasets/PolypDataset.py 위치 변경 후

__init__파일에 

from .PolypDataset import PolypDataset 추가

__all__ = [
    'CustomDataset', 'XMLDataset', 'CocoDataset', 'DeepFashionDataset',
    'VOCDataset', 'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset',
    'LVISV1Dataset', 'GroupSampler', 'DistributedGroupSampler',
    'DistributedSampler', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'ClassBalancedDataset', 'WIDERFaceDataset', 'DATASETS', 'PIPELINES',
    'build_dataset', 'replace_ImageToTensor', 'get_loading_pipeline',
    'NumClassCheckHook', 'CocoPanopticDataset', 'MultiImageMixDataset',
    'OpenImagesDataset', 'OpenImagesChallengeDataset', 'Objects365V1Dataset',
    'Objects365V2Dataset', 'OccludedSeparatedCocoDataset', 'PolypDataset'
]

변경 후 (제일 마지막에 PolypDataset 추가)

python setup.py install


# custum_config.py
csv -> txt 로 변경 후 사용하는 학습용 config파일 제작용 py파일

이 코드로 생성된 코드가 custum_config_save.yaml

해당 코드를 불러 cfg = Config.fromfile('./custum_config_save.yaml')로 모델 불러오기

# train.py
학습을 시키기 위한 코드

# mAP50.py
학습 후 mAP50을 찍어 볼 수 있는 코드
