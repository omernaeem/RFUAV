"""

"""
import torch
import numpy as np
import torch.nn as nn
from utils.trainer import model_init_
from utils.build import check_cfg, build_from_cfg


class Model(nn.Module):
    """
    两个方法，一个benchmark，一个inference
    """
    def __init__(self,
                 cfg:str='../configs/exp1_test.yaml',
                 data_path='../exp_log/',
                 weight_path=''
                 ):
        """
        从cfg中初始化模型
        实现对一个文件夹下的所有图像进行检测
        实现对指定图像进行检测
        实现对某一组原始数据进行检测
        实现对benchmark上的所有数据的推理
        """
        super().__init__()
        if check_cfg(cfg):
            self.cfg = build_from_cfg(cfg)
        self.model, self.classes2index = self.load_model

    def inference(self, image_path):
        """
        :param image_path:
        :return:
        """

    def benchmark(self, data_path):
        """
        对benchmark数据进行推理，并计算指标
        对不同信噪比下的评估集分别进行评估
        :param data_path:
        :return:
        """

    @property
    def load_model(self):
        """
        加载预训练的权重
        :return:
        """

        return model_init_(self.cfg.model, self.cfg.num_class)

    def save_res(self, res_path):
        """
        保存检测结果
        :param res_path:
        :return:
        """


def main():
    test = Model(cfg='./configs/exp1_test.yaml')
    test.inference()


if __name__ == '__main__':
    main()