"""
评估模型在测试集和验证集上的表现
loss曲线，acc曲线
如果metrics写太多的话就把plots的移到这儿
"""


def plot_loss(train_loss, val_loss, epoch, save_path):
    """
    绘制loss曲线
    :param train_loss: 训练集loss
    :param val_loss: 验证集loss
    :param epoch: 训练轮数
    :param save_path: 保存路径
    :return:
    """


def plot_acc(train_acc, val_acc, epoch, save_path):
    """
    绘制acc曲线
    :param train_acc: 训练集acc
    :param val_acc: 验证集acc
    :param epoch: 训练轮数
    :param save_path: 保存路径
    :return:
    """


def confusion_matrix(output, target, num_classes):
    """
    混淆矩阵
    :param output: 预测值
    :param target: 真实值
    :param num_classes: 类别数
    :return:
    """