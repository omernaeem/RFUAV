"""
模型评估的基础参数和最大接口
---------ToDo
加logger和注释
"""

from abc import ABCMeta, abstractmethod
import logging
import torch
from typing import Any, Dict, List, Optional
from torch import Tensor
from ..logger import colorful_logger

class BaseMetric(metaclass=ABCMeta):
    """Base class for metric. original: https://github.com/open-mmlab/mmeval

    To implement a metric, you should implement a subclass of ``BaseMetric``
    that overrides the ``add`` and ``compute_metric`` methods. ``BaseMetric``
    will automatically complete the distributed synchronization between
    processes.

    In the evaluation process, each metric will update ``self._results`` to
    store intermediate results after each call of ``add``. When computing the
    final metric result, the ``self._results`` will be synchronized between
    processes.
    """

    def __init__(self,
                 dataset_meta: Optional[Dict] = None,
                 dist_collect_mode: str = 'unzip',
                 ):
        self.dataset_meta = dataset_meta
        assert dist_collect_mode in ('cat', 'unzip')
        self.dist_collect_mode = dist_collect_mode
        self._results: List[Any] = []

    @property
    def dataset_meta(self) -> Optional[Dict]:
        """Meta information of the dataset."""
        if self._dataset_meta is None:
            return self._dataset_meta
        else:
            return self._dataset_meta.copy()

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: Optional[Dict]) -> None:
        """Set the dataset meta information to the metric."""
        if dataset_meta is None:
            self._dataset_meta = dataset_meta
        else:
            self._dataset_meta = dataset_meta.copy()

    @property
    def name(self) -> str:
        """The metric name, defaults to the name of the class."""
        return self.__class__.__name__

    def reset(self) -> None:
        """Clear the metric stored results."""
        self._results.clear()

    def __call__(self, *args, **kwargs) -> Dict:
        """Stateless call for a metric compute."""
        cache_results = self._results
        self._results = []
        self.add(*args, **kwargs)
        metric_result = self.compute_metric(self._results)
        self._results = cache_results
        return metric_result


    @abstractmethod
    def add(self, *args, **kwargs):
        """Override this method to add the intermediate results to
        ``self._results``.

        Note:
            For performance issues, what you add to the ``self._results``
            should be as simple as possible. But be aware that the intermediate
            results stored in ``self._results`` should correspond one-to-one
            with the samples, in that we need to remove the padded samples for
            the most accurate result.
        """

    @abstractmethod
    def compute_metric(self, results: List[Any]) -> Dict:
        """Override this method to compute the metric result from collectd
        intermediate results.

        The returned result of the metric compute should be a dictionary.
        """


class EVAMetric:
    def __new__(self,
                preds: Tensor,
                labels: Tensor,
                tasks=None,
                num_classes: Optional[int] = None,
                topk: tuple[int, ...] = None,
                save_path: Optional[str] = None,
                classes_name: Optional[tuple[str]] = None,):

        if tasks is None:
            tasks = ['f1', 'precision', 'CM']
        self.logger = self.set_logger
        res = {}

        for task in tasks:

            if task == 'f1':
                from f1 import F1Score
                _preds = []
                for _ in preds:
                    _preds.append(_.argmax())
                preds = [torch.tensor(_preds)]
                f1 = F1Score(num_classes=num_classes, mode=['macro', 'micro'])
                for pred, label in zip(preds, labels):
                    f1.add([pred], [label])

                res['f1'] = f1.compute_metric(f1._results)

            # ToDo
            if task == 'precision':
                from precision import AveragePrecision
                ap = AveragePrecision()
                res['mAP'] = ap(preds, labels)

            if task == 'CM':
                from confusionmatrix import ConfusionMatrix
                cm = ConfusionMatrix(nc=5)
                cm.process_cls_preds(preds, labels)
                for _ in True, False:
                    cm.plot(normalize=_, save_dir=save_path, names=classes_name)

        from topk import Accuracy
        acc = Accuracy(topk=(1, 2, 3))
        res['Top-k'] = acc(preds, labels)

        return res

    @property
    def set_logger(self):
        logger = colorful_logger('Evaluate')
        return logger


# Usage-------------------------------------
def main():
    preds = torch.tensor([[0, 0.9, 0.8, 0.3, 0.6],
                         [0, 0.9, 0.8, 0.7, 0.6],
                         [0, 0.9, 0.8, 0.7, 0.5],
                         [0, 0.9, 0.8, 0.1, 0.6],
                         [0, 0.9, 0.8, 0.7, 0.4]])
    labels = torch.tensor([1, 3, 4, 1, 2])
    metric = EVAMetric(num_classes=5,
                       task=['f1', 'precision'],
                       topk=(1, 3, 5))
    # {'precision': 33.3333, 'recall': 16.6667, 'f1-score': 22.2222}
    metric = EVAMetric(task="multilabel", average='micro', num_classes=3)
    metric(preds, labels)
    # {'precision_micro': 25.0, 'recall_micro': 25.0, 'f1-score_micro': 25.0}


if __name__ == '__main__':
    main()