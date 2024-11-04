"""
dataloader
数据预处理
读标签和种类到内存，画一张数据集情况的图

把数据集和模型hyp设计在CFG中
模型的built以及dataloader的重构
还有一些hpy的重构
"""

from utils.trainer import CustomTrainer

def main():
    model = CustomTrainer(cfg='./configs/exp1_test.yaml')
    model.train()


if __name__ == '__main__':
    main()