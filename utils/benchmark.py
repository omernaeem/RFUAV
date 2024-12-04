import torch
import torch.nn as nn
from utils.trainer import model_init_
from utils.build import check_cfg, build_from_cfg
import os
import glob
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont
import time
from graphic.RawDataProcessor import generate_images
import imageio
import sys
import cv2
from DetModels import YOLOV5S
from DetModels.yolo.basic import LoadImages, Profile, Path, non_max_suppression, Annotator, scale_boxes, colorstr, Colors

# Current directory and metric directory
current_dir = os.path.dirname(os.path.abspath(__file__))
METRIC = os.path.join(current_dir, './metrics')

sys.path.append(METRIC)
sys.path.append(current_dir)
sys.path.append('utils/DetModels/yolo')

from metrics.base_metric import EVAMetric
from torch.utils.data import DataLoader
from logger import colorful_logger

# Supported image and raw data extensions
image_ext = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
raw_data_ext = ['.iq', '.dat']


class Classify_Model(nn.Module):
    """分类阶段的模型切换公共接口
    Performing inference and benchmarking using a pre-trained model.

    Attributes:
    - logger (colorful_logger): Logger for logging messages with color.
    - cfg (str): Path to configuration dictionary.
    - device (str): Device to use for inference (CPU or GPU).
    - model (torch.nn.Module): Pre-trained model.
    - save_path (str): Path to save the results.
    - save (bool): Flag to indicate whether to save the results.
    """

    def __init__(self,
                 cfg: str = '../configs/exp1_test.yaml',
                 weight_path: str = '../default.path',
                 save: bool = True,
                 ):

        super().__init__()
        self.logger = self.set_logger

        if check_cfg(cfg):
            self.logger.log_with_color(f"Using config file: {cfg}")
            self.cfg = build_from_cfg(cfg)

        if self.cfg['device'] == 'cuda':
            if torch.cuda.is_available():
                self.logger.log_with_color("Using GPU for inference")
                self.device = self.cfg['device']
        else:
            self.logger.log_with_color("Using CPU for inference")
            self.device = "cpu"

        if os.path.exists(weight_path):
            self.logger.log_with_color(f"Using weight file: {weight_path}")
            self.weight_path = weight_path
        else:
            raise FileNotFoundError(f"weight path: {weight_path} does not exist")

        self.model = self.load_model
        self.model.to(self.device)
        self.model.eval()
        self.save_path = None

        self.save = save

    def inference(self, source='../example/', save_path: str = '../result'):
        """
        Performs inference on the given source data.

        Parameters:
        - source (str): Path to the source data.
        - save_path (str): Path to save the results.
        """

        if self.save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_path = save_path
            self.logger.log_with_color(f"Saving results to: {save_path}")

        if not os.path.exists(source):
            self.logger.log_with_color(f"Source {source} dose not exit")

        # dir detect
        if os.path.isdir(source):
            data_list = glob.glob(os.path.join(source, '*'))

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
        elif is_valid_file(source, image_ext):
            self.ImgProcessor(source)

        # detect single pack of raw data
        elif is_valid_file(source, raw_data_ext):
            self.RawdataProcess(source)

    # ToDo 给外部用的接口
    def _inference(self, img):
        name = os.path.basename(img)[:-4]
        origin_image = Image.open(img).convert('RGB')
        preprocessed_image = self.preprocess(img)

        temp = self.model(preprocessed_image)

        probabilities = torch.softmax(temp, dim=1)

        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)

        return

    @property
    def load_model(self):
        """
        Loads the pre-trained model.

        Returns:
        - model (torch.nn.Module): Loaded model.
        """

        self.logger.log_with_color(f"Using device: {self.device}")
        model = model_init_(self.cfg['model'], self.cfg['num_classes'], pretrained=True)

        if os.path.exists(self.weight_path):
            self.logger.log_with_color(f"Loading init weights from: {self.weight_path}")
            state_dict = torch.load(self.weight_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.logger.log_with_color(f"Successfully loaded pretrained weights from: {self.weight_path}")
        else:
            self.logger.log_with_color(f"init weights file not found at: {self.weight_path}. Skipping weight loading.")

        return model

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

        temp = self.model(preprocessed_image)

        probabilities = torch.softmax(temp, dim=1)

        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)

        end_time = time.time()
        self.logger.log_with_color(f"Inference time: {(end_time - start_time) / 100 :.8f} sec")
        self.logger.log_with_color(f"{source} contains Drone: {predicted_class_name}, "
                                   f"confidence1: {probabilities[0][predicted_class_index].item() * 100 :.2f} %,"
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

    @property
    def set_logger(self):
        """
        Sets up the logger.

        Returns:
        - logger (colorful_logger): Logger instance.
        """
        logger = colorful_logger('Inference')
        return logger

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

    def benchmark(self, data_path):
        """ToDo
        对benchmark数据进行推理，并计算指标
        对不同信噪比下的评估集分别进行评估
        :param data_path:
        :return:
        """
        self.model.eval()
        _dataset = datasets.ImageFolder(root=data_path, transform=transforms.Compose([
            transforms.Resize((self.cfg['image_size'], self.cfg['image_size'])),
            transforms.ToTensor(),
        ]))
        dataset = DataLoader(_dataset, batch_size=self.cfg['batch_size'], shuffle=self.cfg['shuffle'])
        self.logger.log_with_color("Starting Benchmarking...")

        loss = 0.0
        correct = 0
        total = 0
        probabilities = []
        total_labels = []
        classes_name = tuple(self.cfg['class_names'].keys())

        for images, labels in dataset:
            images, labels = images.to(self.cfg['device']), labels.to(self.cfg['device'])
            outputs = self.model(images)
            for output in outputs:
                probabilities.append(list(torch.softmax(output, dim=0)))
            loss += nn.CrossEntropyLoss(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_labels.append(labels)
        _total_labels = torch.concat(total_labels, dim=0)
        _probabilities = torch.tensor(probabilities)

        metrics = EVAMetric(preds=_probabilities.to(self.cfg['device']),
                            labels=_total_labels,
                            num_classes=self.cfg['num_classes'],
                            tasks=('f1', 'precision', 'CM'),
                            topk=(1, 3, 5),
                            save_path=data_path,
                            classes_name=classes_name)

        metrics['acc'] = 100 * correct / total
        metrics['total_loss'] = loss / len(dataset)
        return metrics


class Detection_Model:
    """做一个调模型的公共接口

    """
    def __init__(self, cfg):
        """做一个解析cfg的功能，然后按照cfg指定选模型

        :param cfg:

        """
        model_name = cfg['model_name']
        weight_path = cfg['weight_path']
        if model_name == 'yolov5':
            self.S1model = YOLOV5S(weights=cfg)
            self.S1model.inference = self.yolov5_detect

        # ToDo
        elif model_name == 'faster_rcnn':
            self.S1model = YOLOV5S(weights=cfg)
            self.S1model.inference = self.yolov5_detect

    #ToDo 给外部用的接口
    def yolov5_detect(self,
                      source='../example/source/',
                      save_dir='../res',
                      weight_path='../example/detect/',
                      imgsz=(640, 640),
                      conf_thres=0.25,
                      iou_thres=0.45,
                      max_det=1000,
                      line_thickness=3,
                      hide_labels=False,
                      hide_conf=False,
                      ):

        color = Colors()
        detmodel = self.S1model
        stride, names = detmodel.stride, detmodel.names
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Run inference
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, s in dataset:
            im = torch.from_numpy(im).to(detmodel.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = detmodel(im)
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, max_det=max_det)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                res = {}  # Drone
                p = Path(p)  # to Path
                save_path = str(save_dir + p.name)  # im.jpg
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        res[names[int(c)]] = int(n)

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        annotator.box_label(xyxy, label, color=color(c+2, True))

                # Stream results
                im0 = annotator.result()
                # Save results (image with detections)
                cv2.imwrite(save_path, im0)

        # Print results
        print(f"Results saved to {colorstr('bold', save_dir)}")

    def faster_rcnn_detect(self,
                           source='../example/source/',
                           save_dir='../res',
                           weight_path='../example/detect/',
                           imgsz=(640, 640),
                           conf_thres=0.25,
                           iou_thres=0.45,
                           max_det=1000,
                           line_thickness=3,
                           hide_labels=False,
                           hide_conf=False,
    ):
        pass


def is_valid_file(path, total_ext):
    """
    Checks if the file has a valid extension.

    Parameters:
    - path (str): Path to the file.
    - total_ext (list): List of valid extensions.

    Returns:
    - bool: True if the file has a valid extension, False otherwise.
    """
    last_element = os.path.basename(path)
    if any(last_element.lower().endswith(ext) for ext in total_ext):
        return True
    else:
        return False


def get_key_from_value(d, value):
    """
    Gets the key from a dictionary based on the value.

    Parameters:
    - d (dict): Dictionary.
    - value: Value to find the key for.

    Returns:
    - key: Key corresponding to the value, or None if not found.
    """
    for key, val in d.items():
        if val == value:
            return key
    return None


# Usage-----------------------------------------------------------------------------------------------------------------
def main():

    """
    test = Model(cfg='E:/Train_log/RFUAV/exp1_test/config.yaml',
                 weight_path='E:/Train_log/RFUAV/exp1_test/ResNet_epoch_29.pth')

    test.inference(source='E:/FowardRes_log/RFUAV/1.code_check/exp1/source',
                   save_path='E:/FowardRes_log/RFUAV/1.code_check/exp1/res')
    # test.benchmark()
    """
    test = Detection_Model('test')
    test.yolov5_detect(source='C:/Users/user/Desktop/ceshi/',
                   save_dir='C:/ML/RFUAV/res/',
                   weight_path='E:/Drone_dataset/RFUAV/yolotest/weights/best.pt'
                   )

if __name__ == '__main__':
    main()