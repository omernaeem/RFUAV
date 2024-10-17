"""
直接把CFG中的组件组一个模型
"""
import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from mmengine.config import Config, ConfigDict
from mmengine.utils import ManagerMixin

if TYPE_CHECKING:
    import torch.nn as nn

    from mmengine.optim.scheduler import _ParamScheduler
    from mmengine.runner import Runner


def build_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    If the global variable default scope (:obj:`DefaultScope`) exists,
    :meth:`build` will firstly get the responding registry and then call
    its own :meth:`build`.

    At least one of the ``cfg`` and ``default_args`` contains the key "type",
    which should be either str or class. If they all contain it, the key
    in ``cfg`` will be used because ``cfg`` has a high priority than
    ``default_args`` that means if a key exists in both of them, the value of
    the key will be ``cfg[key]``. They will be merged first and the key "type"
    will be popped up and the remaining keys will be used as initialization
    arguments.

    Examples:
        >>> from mmengine import Registry, build_from_cfg
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     def __init__(self, depth, stages=4):
        >>>         self.depth = depth
        >>>         self.stages = stages
        >>> cfg = dict(type='ResNet', depth=50)
        >>> model = build_from_cfg(cfg, MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict or ConfigDict or Config): Config dict. It should at least
            contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. Defaults to None.

    Returns:
        object: The constructed object.
    """
    # Avoid circular import
    from ..logging import print_log

    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')

    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be a mmengine.Registry object, '
                        f'but got {type(registry)}')

    if not (isinstance(default_args,
                       (dict, ConfigDict, Config)) or default_args is None):
        raise TypeError(
            'default_args should be a dict, ConfigDict, Config or None, '
            f'but got {type(default_args)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    # Instance should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.scope}::{registry.name} registry. '  # noqa: E501
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected. More details '
                    'can be found at '
                    'https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module'  # noqa: E501
                )
        # this will include classes, functions, partial functions and more
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        # If `obj_cls` inherits from `ManagerMixin`, it should be
        # instantiated by `ManagerMixin.get_instance` to ensure that it
        # can be accessed globally.
        if inspect.isclass(obj_cls) and \
                issubclass(obj_cls, ManagerMixin):  # type: ignore
            obj = obj_cls.get_instance(**args)  # type: ignore
        else:
            obj = obj_cls(**args)  # type: ignore

        if (inspect.isclass(obj_cls) or inspect.isfunction(obj_cls)
                or inspect.ismethod(obj_cls)):
            print_log(
                f'An `{obj_cls.__name__}` instance is built from '  # type: ignore # noqa: E501
                'registry, and its implementation can be found in '
                f'{obj_cls.__module__}',  # type: ignore
                logger='current',
                level=logging.DEBUG)
        else:
            print_log(
                'An instance is built from registry, and its constructor '
                f'is {obj_cls}',
                logger='current',
                level=logging.DEBUG)
        return obj


def build_model_from_cfg(
   cfg: Union[dict, ConfigDict, Config],
   default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
) -> 'nn.Module':
   """Build a PyTorch model from config dict(s). Different from
   ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

   Args:
      cfg (dict, list[dict]): The config of modules, which is either a config
         dict or a list of config dicts. If cfg is a list, the built
         modules will be wrapped with ``nn.Sequential``.
      registry (:obj:`Registry`): A registry the module belongs to.
      default_args (dict, optional): Default arguments to build the module.
         Defaults to None.

   Returns:
      nn.Module: A built nn.Module.
   """
   from .model import Sequential
   if isinstance(cfg, list):
      modules = [
         build_from_cfg(_cfg, registry, default_args) for _cfg in cfg
      ]
      return Sequential(*modules)
   else:
      return build_from_cfg(cfg, registry, default_args)


def build_scheduler_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, ConfigDict, Config]] = None
) -> '_ParamScheduler':
    """Builds a ``ParamScheduler`` instance from config.

    ``ParamScheduler`` supports building instance by its constructor or
    method ``build_iter_from_epoch``. Therefore, its registry needs a build
    function to handle both cases.

    Args:
        cfg (dict or ConfigDict or Config): Config dictionary. If it contains
            the key ``convert_to_iter_based``, instance will be built by method
            ``convert_to_iter_based``, otherwise instance will be built by its
            constructor.
        registry (:obj:`Registry`): The ``PARAM_SCHEDULERS`` registry.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. It must contain key ``optimizer``. If
            ``convert_to_iter_based`` is defined in ``cfg``, it must
            additionally contain key ``epoch_length``. Defaults to None.

    Returns:
        object: The constructed ``ParamScheduler``.
    """
    assert isinstance(
        cfg,
        (dict, ConfigDict, Config
         )), f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}'
    assert isinstance(
        registry, Registry), ('registry should be a mmengine.Registry object',
                              f'but got {type(registry)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        convert_to_iter = args.pop('convert_to_iter_based', False)
        if convert_to_iter:
            scheduler_type = args.pop('type')
            assert 'epoch_length' in args and args.get('by_epoch', True), (
                'Only epoch-based parameter scheduler can be converted to '
                'iter-based, and `epoch_length` should be set')
            if isinstance(scheduler_type, str):
                scheduler_cls = registry.get(scheduler_type)
                if scheduler_cls is None:
                    raise KeyError(
                        f'{scheduler_type} is not in the {registry.name} '
                        'registry. Please check whether the value of '
                        f'`{scheduler_type}` is correct or it was registered '
                        'as expected. More details can be found at https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module'  # noqa: E501
                    )
            elif inspect.isclass(scheduler_type):
                scheduler_cls = scheduler_type
            else:
                raise TypeError('type must be a str or valid type, but got '
                                f'{type(scheduler_type)}')
            return scheduler_cls.build_iter_from_epoch(  # type: ignore
                **args)
        else:
            args.pop('epoch_length', None)
            return build_from_cfg(args, registry)



def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset"""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)


def build_transforms(self, hyp=None):
    """Builds and appends transforms to the list."""
    if self.augment:
        hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
        hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
        transforms = v8_transforms(self, self.imgsz, hyp)
    else:
        transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
    transforms.append(
        Format(bbox_format='xywh',
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask))
    return transforms