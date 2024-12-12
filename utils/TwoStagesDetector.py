import os
import glob

import torch
from PIL import Image
import cv2
from graphic.RawDataProcessor import waterfall_spectrogram
from logger import colorful_logger
import json
import numpy as np
from benchmark import Classify_Model, Detection_Model, is_valid_file, raw_data_ext, image_ext


class TwoStagesDetector:

    def __init__(self, cfg: str = ''):

        """A data flow processing class for a two-stage model, providing public interfaces.

        Args:
            cfg (str): Path to the configuration file.
        """

        self.logger = colorful_logger('Inference')
        det, cla, save_path, target_dir = load_model_from_json(cfg)
        self.det = det
        self.cla = cla
        self.save_path = save_path
        self.target_dir = target_dir

        if not cla and det:
            self.DroneDetector(cfg=det)
        elif not det and cla:
            self.DroneClassifier(cfg=cla['cfg'], weight_path=cla['weight_path'], save=True)
        elif det and cla:
            self.DroneDetector(cfg=det)
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

    def ImgProcessor(self, source, save=True):

        """Processes an image source using the first and second stage models.

        Args:
            source: The image source to be processed.
            save (bool): Whether to save the processed image.

        Returns:
            Processed image if `save` is False, otherwise None.
        """

        if self.S1.S1model:
            if save:
                with torch.no_grad:
                    res = self.S1.S1model.inference(source=source, save_dir=self.target_dir)
            else:
                source.seek(0)
                temp = np.asarray(bytearray(source.read()), dtype=np.uint8)
                temp = cv2.imdecode(temp, cv2.IMREAD_COLOR)
                res = self.S1.S1model.inference(source=temp)
            if not self.S2model:
                if save:
                    cv2.imwrite(self.save_path, res)
                else:
                    return res

        if self.S2model:
            if save: name = os.path.basename(source)[:-4]
            origin_image = Image.open(source).convert('RGB')
            preprocessed_image = self.S2model.preprocess(source)

            probability, predicted_class_name = self.S2model.forward(preprocessed_image)

            if not self.S1.S1model:
                res = self.S2model.add_result(res=predicted_class_name,
                                              probability=predicted_class_name,
                                              image=origin_image)
                if save:
                    res.save(os.path.join(self.save_path, name + '.jpg'))
                else:
                    return res

            else:
                res = put_res_on_img(res, predicted_class_name, probability=probability)

                if save:
                    cv2.imwrite(self.save_path, res)
                else:
                    return res

    def RawdataProcess(self, source):

        """
        Transforming raw data into a video and performing inference on video.

        Parameters:
        - source (str): Path to the raw data.
        """

        test_times = 0
        images = waterfall_spectrogram(source, fft_size=256, fs=100e6, location='buffer', time_scale=39062)
        name = os.path.splitext(os.path.basename(source))
        with torch.no_grad():
            while images:
                test_times += 1
                _ = self.ImgProcessor(images[0], save=False)

                if test_times == 1:
                    height, width, layers = _.shape
                    video_name = name[0] + '_output.avi'
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video = cv2.VideoWriter(os.path.join(self.save_path, video_name), fourcc, 30, (width, height))
                video.write(_)
                del images[0]
        video.release()
        self.logger.log_with_color(f"Finished processing {name[0]}.")

    def DroneDetector(self, cfg):

        """Initializes the first stage model.

        Args:
            cfg: Configuration for the detector model.
        """

        self.S1 = Detection_Model(cfg)

    def DroneClassifier(self, cfg, weight_path, save=True):

        """Initializes the second stage model.

        Args:
            cfg: Configuration for the classifier model.
            weight_path: Path to the weights for the classifier model.
            save (bool): Whether to save the model.
        """

        self.S2model = Classify_Model(cfg=cfg, weight_path=weight_path)

    @property
    def set_logger(self):

        """
        Sets up the logger.

        Returns:
        - logger (colorful_logger): Logger instance.
        """

        logger = colorful_logger('Inference')
        return logger


def load_model_from_json(cfg):

    """Loads configuration from a JSON file.

    Args:
        cfg (str): Path to the configuration file.

    Returns:
        Tuple containing detector configuration, classifier configuration, save path, and target directory.
    """

    with open(cfg, 'r') as f:
        _ = json.load(f)
        return _['detector'] if 'detector' in _ else None, _['classifier'] if 'classifier' in _ else None, _['save_dir'], _['target_dir']


def put_res_on_img(img,
                   text,
                   probability=0.0,
                   position=(20, 60),
                   font_scale=1,
                   color=(0, 0, 0),
                   thickness=3):

    """Adds text result on an image.

    Args:
        img: Image to add text to.
        text (str): Text to add.
        probability (float): Probability value to display alongside the text.
        position (tuple): Position of the text on the image.
        font_scale (int): Font scale of the text.
        color (tuple): Color of the text.
        thickness (int): Thickness of the text.

    Returns:
        Image with added text.
    """

    # 在图片上添加文字
    cv2.putText(img=img,
                text=text + f" {probability:.2f}%",
                org=position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA)

    return img


# for test ------------------------------------------------------------------------------------------------------------
def main():
    cfg_path = '../example/two_stage/sample.json'
    TwoStagesDetector(cfg=cfg_path)


if __name__ == '__main__':
    main()