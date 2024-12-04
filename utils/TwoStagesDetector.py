import torch
import os
import glob
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import time
from graphic.RawDataProcessor import generate_images
import imageio
from logger import colorful_logger
import json

from benchmark import Classify_Model, Detection_Model, get_key_from_value, is_valid_file, raw_data_ext, image_ext


# 二阶段模型的一个数据流处理类，提供公共接口
class TwoStagesDetector:

    def __init__(self, cfg: str = ''):
        """解析一个统一cfg的方法，并按照方法选模型
        1.选设备
        2.解析cfg，初始化模型
        """
        self.logger = colorful_logger('Inference')
        det, cla, save_path, target_dir = self.load(cfg)
        self.det = det
        self.cla = cla
        self.save_path = save_path
        self.target_dir = target_dir

        if not cla and det:
            self.DroneDetector()
        elif not det and cla:
            self.DroneClassifier(cfg=cla['cfg'], weight_path=cla['weight_path'], save=True)
        elif det and cla:
            self.DroneDetector(cfg='')
            self.DroneClassifier(cfg=cla['cfg'], weight_path=cla['weight_path'], save=True)
        else:
            raise ValueError("No model is selected")


        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.logger.log_with_color(f"Saving results to: {save_path}")

        if not os.path.exists(target_dir):
            raise ValueError(f"Source {target_dir} dose not exit")

        # dir detect
        if os.path.isdir(target_dir):
            data_list = glob.glob(os.path.join(target_dir, '*'))

            for data in data_list:
                # detect images in dir
                if is_valid_file(data, image_ext):
                    self.ImgProcessor(data)
                # detect raw datas in dir
                elif is_valid_file(data, raw_data_ext):
                    self.RawdataProcess(data)
                else:
                    continue

        # detect single image
        elif is_valid_file(target_dir, image_ext):
            self.ImgProcessor(target_dir)

        # detect single pack of raw data
        elif is_valid_file(target_dir, raw_data_ext):
            self.RawdataProcess(target_dir)

    def load(self, cfg):
        """load cfg from .json
        :return:
        """
        with open(cfg, 'r') as f:
            _ = json.load(f)
            return _['detector'] if 'detector' in _ else None, _['classifier'] if 'classifier' in _ else None, _['target_dir'], _['save_dir']

    def ImgProcessor(self, source):
        """
         Performs inference on spectromgram data.

        Parameters:
        - source (str): Path to the image.
        """

        start_time = time.time()

        name = os.path.basename(source)[:-4]
        origin_image = Image.open(source).convert('RGB')
        preprocessed_image = self.preprocess(source)

        temp = self.DroneClassifier.inference(preprocessed_image)

        probabilities = torch.softmax(temp, dim=1)

        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)

        end_time = time.time()
        self.logger.log_with_color(f"Inference time: {(end_time - start_time) / 100 :.8f} sec")
        self.logger.log_with_color(f"{source} contains Drone: {predicted_class_name}, "
                                   f"confidence: {probabilities[0][predicted_class_index].item() * 100 :.2f} %,"
                                   f" start saving result")

        if self.save:
            res = self.add_result(res=predicted_class_name,
                                  probability=probabilities[0][predicted_class_index].item() * 100,
                                  image=origin_image)

            res.save(os.path.join(self.save_path, name + '.jpg'))

    def RawdataProcess(self, source):
        """
        Transforming raw data into a video and performing inference on video.

        Parameters:
        - source (str): Path to the raw data.
        """
        res = []
        images = generate_images(source)
        name = os.path.splitext(os.path.basename(source))

        for image in images:
            temp = self.model(self.preprocess(image))

            probabilities = torch.softmax(temp, dim=1)

            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)

            _ = self.add_result(res=predicted_class_name,
                                probability=probabilities[0][predicted_class_index].item() * 100,
                                image=image)
            res.append(_)

        imageio.mimsave(os.path.join(self.save_path, name + '.mp4'), res, fps=5)

    def add_result(self,
                   res,
                   image,
                   position=(40, 40),
                   font="arial.ttf",
                   font_size=45,
                   text_color=(255, 0, 0),
                   probability=0.0
                   ):
        """
        Adds the inference result to the image.

        Parameters:
        - res (str): Inference result.
        - image (PIL.Image): Input image.
        - position (tuple): Position to add the text.
        - font (str): Font file path.
        - font_size (int): Font size.
        - text_color (tuple): Text color.
        - probability (float): Confidence probability.

        Returns:
        - image (PIL.Image): Image with added result.
        """
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font, font_size)
        draw.text(position, res + f" {probability:.2f}%", fill=text_color, font=font)

        return image

    def preprocess(self, img):

        transform = transforms.Compose([
            transforms.Resize((self.cfg['image_size'], self.cfg['image_size'])),
            transforms.ToTensor(),
        ])

        image = Image.open(img).convert('RGB')
        preprocessed_image = transform(image)

        preprocessed_image = preprocessed_image.to(self.device)
        preprocessed_image = preprocessed_image.unsqueeze(0)

        return preprocessed_image

    def DroneDetector(self, cfg):
        """单独的发现流程

        :return:
        """
        self.S1 = Detection_Model(cfg='E:/Drone_dataset/RFUAV/yolotest/weights/best.pt')
        """
        self.S1.S1model.inference(source='C:/Users/user/Desktop/ceshi/',
                   save_dir='C:/ML/RFUAV/res/',
                   )
        """


    def DroneClassifier(self, cfg, weight_path, save=True):
        """#单独的分类流程

        :return:
        """
        self.S2model = Classify_Model(cfg=cfg, weight_path=weight_path, save=save)
        # self.S2model._inference()


    @property
    def set_logger(self):
        """
        Sets up the logger.

        Returns:
        - logger (colorful_logger): Logger instance.
        """
        logger = colorful_logger('Inference')
        return logger


# for test ------------------------------------------------------------------------------------------------------------
def main():
    cfg_path = '../example/two_stage/sample.json'
    TwoStagesDetector(cfg=cfg_path)


if __name__ == '__main__':
    main()