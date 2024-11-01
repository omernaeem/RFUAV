"""build from CFG
"""
import yaml
import logging
import os
import torch

DefaultConfig = '../configs/config.yaml'
Model_list = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2",
    "mobilenet_v2", "shufflenet_v2_x0_"]


def check_cfg(cfg: str):
    opt = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
    if os.path.exists(opt.train):
        raise ValueError("Training data path does not exist: {}".format(opt.train))
    if os.path.exists(opt.val):
        raise ValueError("Validation data path does not exist: {}".format(opt.val))
    if not isinstance(opt.model, str) or opt.model.lower() not in Model_list:
        raise ValueError("The model you specified is not available")
    if not isinstance(opt.num_classes, int):
        raise ValueError("The number of classes must be an integer")
    if opt.device == 0 and not torch.cuda.is_available():
        logging.info("CUDA is not available, using CPU instead")
        opt.device = "cpu"
    return True



def build_from_cfg(cfg: str = DefaultConfig):
    if cfg != DefaultConfig:
        if not check_cfg(cfg):
            raise ValueError("Invalid config file: {}".format(cfg))
        logging.info("Using custom config: {}".format(cfg))
    opt = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)

    return opt