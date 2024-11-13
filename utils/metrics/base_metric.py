"""
模型评估的基础参数和最大接口
---------ToDo
加logger和注释
"""

from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import Union, Sequence
import torch
from precision import MultiLabelMixin
from typing import Any, Dict, List, Optional


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


class EVAlMetric(MultiLabelMixin, BaseMetric):
    """Wrapper to get different task of PrecisionRecallF1score calculation, by
    setting the ``task`` argument to either ``'singlelabel'`` or
    ``multilabel``.

    See the documentation of :mod:`SingleLabelPrecisionRecallF1score` and
    :mod:`MultiLabelPrecisionRecallF1score` for the detailed usages and
    examples.
    """

    def __new__(cls,
                task: str = 'singlelabel',
                num_classes: Optional[int] = None,
                thrs: Union[float, Sequence[Optional[float]], None] = None,
                topk: Optional[int] = None,
                items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                average: Optional[str] = 'macro',
                **kwargs):

        assert isinstance(thrs, float) or thrs is None, \
            "task `'multilabel'` only supports single threshold or None."
        assert isinstance(num_classes, int), \
            '`num_classes` is necessary for multi-label metrics.'
        return MultiLabelPrecisionRecallF1score(
            num_classes=num_classes,
            thr=thrs,
            topk=topk,
            items=items,
            average=average,
            **kwargs)


# Usage-------------------------------------
def main():

    preds = torch.tensor([2, 0, 1, 1])
    labels = torch.tensor([2, 1, 2, 0])
    metric = EVAlMetric(num_classes=3)
    metric(preds, labels)
    # {'precision': 33.3333, 'recall': 16.6667, 'f1-score': 22.2222}
    metric = EVAlMetric(task="multilabel", average='micro', num_classes=3)
    metric(preds, labels)
    # {'precision_micro': 25.0, 'recall_micro': 25.0, 'f1-score_micro': 25.0}


if __name__ == '__main__':
    main()