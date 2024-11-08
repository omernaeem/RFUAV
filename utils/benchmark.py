import torch
import numpy as np
import torch.nn as nn
from utils.trainer import model_init_
from utils.build import check_cfg, build_from_cfg
import os
import logging
import glob
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import time

image_ext = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
raw_data_ext = ['.iq', '.dat']


class Model(nn.Module):

    def __init__(self,
                 cfg: str = '../configs/exp1_test.yaml',
                 weight_path: str = '../default.path',
                 save: bool = True,
                 ):
        """
        从cfg中初始化模型
        实现对一个文件夹下的所有图像进行检测
        实现对指定图像进行检测
        实现对某一组原始数据进行检测
        实现对benchmark上的所有数据的推理
        """
        super().__init__()
        self.logger = self.set_logger

        if check_cfg(cfg):
            self.logger.info(f"Using config file: {cfg}")
            self.cfg = build_from_cfg(cfg)

        if self.cfg.device == 'cuda':
            if torch.cuda.is_available():
                self.logger.info("Using GPU for inference")
                self.device = self.cfg.device
        else:
            self.logger.info("Using CPU for inference")
            self.device = "cpu"

        if os.path.exists(weight_path):
            self.logger.info(f"Using weight file: {weight_path}")
            self.weight_path = weight_path
        else:
            raise FileNotFoundError(f"weight path: {weight_path} does not exist")

        self.model = self.load_model
        self.model.to(self.device)
        self.model.eval()

        self.save = save

    def inference(self, source='../example/', save_path: str = '../result'):

        if self.save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_path = save_path
            self.logger.info(f"Saving results to: {save_path}")

        if not os.path.exists(source):
            self.logger.info(f"Source {source} dose not exit")

        if os.path.isdir(source):
            data_list = glob.glob(os.path.join(source, '*'))

            for data in data_list:

                if is_valid_file(data, image_ext):
                    self.ImgProcessor(data)

                elif is_valid_file(data, raw_data_ext):
                    self.RawdataProcess(data)

                else: continue

        elif is_valid_file(source, image_ext):
            self.ImgProcessor(source)

        elif is_valid_file(source, raw_data_ext):
            self.RawdataProcess(source)

    @property
    def load_model(self):

        self.logger.info(f"Using device: {self.device}")
        model = model_init_(self.cfg.model, self.cfg.num_class, pretrained=self.cfg.pretrained)

        if os.path.exists(self.weight_path):
            self.logger.info(f"Loading init weights from: {self.weight_path}")
            state_dict = torch.load(self.weight_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.logger.info(f"Successfully loaded pretrained weights from: {self.weight_path}")
        else:
            self.logger.warning(f"init weights file not found at: {self.weight_path}. Skipping weight loading.")

        return model

    def ImgProcessor(self, source):

        start_time = time.time()
        self.logger.info(f"Processing image: {source} at {start_time}")

        name = os.path.basename(source)
        origin_image = Image.open(source).convert('RGB')
        preprocessed_image = self.preprocess(source)

        temp = self.model(preprocessed_image)
        output = self.cfg.class_names[temp]

        end_time = time.time()
        self.logger.info(f"Inference time: {end_time - start_time} sec")
        self.logger.info(f"{source} contains Drone: {output}, start saving result")

        if self.save:
            res = self.add_result(output, origin_image)
            res.save(os.path.join(self.save_path), name)

    def RawdataProcess(self, source):
        """ToDo
        仅支持用特定python色标画图的原始数据进行推理
        用python的画图程序把一个原始数据按fps=15画出来，
        将所有图像结果预测出来后变成转换成一个视频
        """

    def add_result(self,
                   res,
                   image,
                   position=(10, 10),
                   font="arial.ttf",
                   font_size=20,
                   text_color=(255, 0, 0)):

            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font, font_size)
            draw.text(position, res, fill=text_color, font=font)
            return image

    @property
    def set_logger(self):

        logger = logging.getLogger("InferenceLogger")
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def preprocess(self, img):

        transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
        ])

        image = Image.open(img).convert('RGB')
        preprocessed_image = transform(image)

        preprocessed_image = preprocessed_image.to(self.device)
        preprocessed_image = preprocessed_image.unsqueeze(0)

        return preprocessed_image

    def benchmark(self, data_path):
        """
        对benchmark数据进行推理，并计算指标
        对不同信噪比下的评估集分别进行评估
        :param data_path:
        :return:
        """


def is_valid_file(path, total_ext):
    last_element = os.path.basename(path)
    if any(last_element.lower().endswith(ext) for ext in total_ext):
        return True
    else:
        return False


def main():

    test = Model(cfg='E:/Train_log/RFUAV/exp1_test/config.yaml',
                 weight_path='E:/Train_log/RFUAV/exp1_test/config.yaml/ResNet_epoch_29.pth')

    test.inference(source='E:/FowardRes_log/RFUAV/1.code_check/exp1/source',
                   save_path='E:/FowardRes_log/RFUAV/1.code_check/exp1/res')
    # test.benchmark()


if __name__ == '__main__':
    main()