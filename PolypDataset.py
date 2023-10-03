import cv2

from mmdet.datasets.custom import CustomDataset
import os.path as osp

from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS

CLASSES = ("polyp",)
cat2label = {k: i for i, k in enumerate(CLASSES)}
@DATASETS.register_module(force=True)
class PolypDataset(CustomDataset):
    CLASSES = ("polyp",)

    def load_annotations(self, ann_file):
        print('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:',
              self.img_prefix)
        print('#### ann_file:', ann_file)
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        image_list = mmcv.list_from_file(self.ann_file)
        # 포맷 중립 데이터를 담을 list 객체
        data_infos = []

        for image_id in image_list:
            filename = '{0:}/{1:}.jpg'.format(self.img_prefix, image_id)
            # 원본 이미지의 너비, 높이를 image를 직접 로드하여 구함.
            image = cv2.imread(filename)
            height, width = image.shape[:2]
            # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename 에는 image의 파일명만 들어감(디렉토리는 제외)
            data_info = {'filename': str(image_id) + '.jpg',
                         'width': width, 'height': height}
            # 개별 annotation이 있는 서브 디렉토리의 prefix 변환.
            label_prefix = self.img_prefix.replace('images', 'bbox')
            # 개별 annotation 파일을 1개 line 씩 읽어서 list 로드
            lines = mmcv.list_from_file(osp.join(label_prefix, str(image_id) + '.txt'))

            # 전체 lines를 개별 line별 공백 레벨로 parsing 하여 다시 list로 저장. content는 list의 list형태임.
            # ann 정보는 numpy array로 저장되나 텍스트 처리나 데이터 가공이 list 가 편하므로 일차적으로 list로 변환 수행.
            content = [line.strip().split(' ') for line in lines]
            # 오브젝트의 클래스명은 bbox_names로 저장.
            bbox_names = [x[0] for x in content]

            # bbox 좌표를 저장
            bboxes = [[float(info) for info in x[1:]] for x in content]

            # 클래스명이 해당 사항이 없는 대상 Filtering out, 'DontCare'는 ignore로 별도 저장.
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            for bbox_name, bbox in zip(bbox_names, bboxes):
                # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가

                if bbox_name in cat2label:
                    gt_bboxes.append(bbox)
                    # gt_labels에는 class id를 입력
                    gt_labels.append(cat2label[bbox_name])
                else:
                    gt_bboxes_ignore.append(bbox)
                    gt_labels_ignore.append(-1)
            # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값은 모두 np.array임.
            data_anno = {
                'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                'labels': np.array(gt_labels, dtype=np.longlong),
                'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array(gt_labels_ignore, dtype=np.longlong)
            }
            # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장.
            data_info.update(ann=data_anno)
            # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
            data_infos.append(data_info)

        return data_infos


    def evaluate(self, results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
