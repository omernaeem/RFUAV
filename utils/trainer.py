from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import os
import logging
from build import build_from_cfg, check_cfg


class Basetrainer:
    def __init__(self,
                 model: str,
                 train_path: str,
                 val_path: str,
                 num_class: int,
                 save_path: str,
                 weight_path: str = "",
                 log_file: str = "train.log",
                 device: str = "cuda",
                 criterion=nn.CrossEntropyLoss(),
                 pretrained: bool = True,
                 batch_size: int = 8,
                 shuffle: bool = False,
                 image_size: int = 224,
                 ):

        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.num_class = num_class
        self.save_path = save_path
        self.best_acc = 0
        self.best_loss = 1e6
        self.best_epoch = 0
        self.best_model = None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = self.set_logger(os.path.join(save_path, log_file))
        self.criterion = criterion  # initalizeing the loss function
        self.set_up(model=model, train_path=train_path, val_path=val_path, pretrained=pretrained,
                    weight_path=weight_path)

    def set_up(self, train_path, val_path, pretrained, weight_path, model='resnet18'):
        """
         Set up the training environment.
         """

        self.logger.info(f"Loading model: {model}")

        if os.path.exists(weight_path):
            pretrained = False

        if not os.path.exists(pretrained):
            self.logger.info("Pretrained model not found, using default weight")
            pretrained = True

        self.model = model_init_(model_name=model, num_class=self.num_class, pretrained=pretrained)
        # resnet serise model

        if os.path.exists(weight_path):
            self.load_pretrained_weights(weight_path)
            self.logger.info(f"Loading pretrained weights from: {weight_path}")

        self.model.to(self.device)
        self.logger.info(f"{model} loaded onto device: {self.device}")

        # initializing the dataset
        self.logger.info(f"Loading dataset from: {train_path} and {val_path}")
        _train_set = datasets.ImageFolder(root=train_path, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        self.train_set = DataLoader(_train_set, batch_size=self.batch_size, shuffle=self.shuffle)

        _val_set = datasets.ImageFolder(root=val_path, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        self.val_set = DataLoader(_val_set, batch_size=self.batch_size, shuffle=self.shuffle)

        # initializing optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            self.model.train()  # 将模型设置为训练模式
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in self.train_set:
                images, labels = images.to(self.device), labels.to(self.device)
                # 清除之前的梯度
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播并优化
                loss.backward()
                self.optimizer.step()

                # 统计训练过程中的损失和准确率
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            train_loss = running_loss / len(self.train_set)
            train_acc = 100 * correct / total
            self.logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
            val_acc, val_loss = self.val()
            self.logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
            self.save_model(val_acc, epoch)

    def val(self):
        self.logger.info("Starting validation...")
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in self.val_set:
                val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                val_outputs = self.model(val_images)
                val_loss += self.criterion(val_outputs, val_labels).item()
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()
        return 100 * val_correct / val_total, val_loss / len(self.val_set)

    def save_model(self, val_acc, epoch):
        """
        Save the model after each epoch and track the best model based on validation accuracy.
        """

        checkpoint_path = os.path.join(self.save_path, f'{self.model._get_name()}_epoch_{epoch + 1}.pth')
        self.logger.info(f'Model saved at {checkpoint_path} (Validation Accuracy: {val_acc:.2f}%)')
        torch.save(self.model.state_dict(), checkpoint_path)

        # Save the best model if current validation accuracy is higher than the best recorded one
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_model = self.model.state_dict()
            best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(self.best_model, best_model_path)
            self.logger.info(f'New best model saved with Accuracy: {val_acc:.2f}%')

    def set_logger(self, log_file):
        """
        Set up the logger to output to both console and a log file.
        """

        # Create a logger
        logger = logging.getLogger("TrainerLogger")
        logger.setLevel(logging.INFO)

        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create file handler and set level to info
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Add formatter to handlers
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger

    def load_pretrained_weights(self, weight_path: str):
        """
        Load pretrained weights from the given path if the file exists.
        """
        if os.path.exists(weight_path):
            self.logger.info(f"Loading pretrained weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info(f"Successfully loaded pretrained weights from: {weight_path}")
        else:
            self.logger.warning(f"Pretrained weights file not found at: {weight_path}. Skipping weight loading.")


def model_init_(model_name, num_class, pretrained=True):
    # resnet serise model
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)

    # ViT serise model
    elif model_name == "vit_b_16":
        model = models.vit_b_16(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name == "vit_b_32":
        model = models.vit_b_32(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name == "vit_l_16":
        model = models.vit_l_16(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name == "vit_l_32":
        model = models.vit_l_32(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name == "vit_h_14":
        model = models.vit_h_14(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)

    # SiwnTrans serise mdoel
    elif model_name == "swin_v2_t":
        model = models.swin_v2_t(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)
    elif model_name == "swin_v2_s":
        model = models.swin_v2_s(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)
    elif model_name == "swin_v2_b":
        model = models.swin_v2_b(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)

    # Mobilnet serise model
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_class)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_class)

    else:
        raise ValueError("model not supported")

    return model


class CustomTrainer(Basetrainer):

    def __init__(self,
                 cfg: str,
                 ):
        if check_cfg(cfg):
            self.parameters = build_from_cfg(cfg)
        else:
            super(Basetrainer).__init__()
        print('test')

    def save_yaml(self):
        """
        把训练中使用的一些预设参数:preprocessing使用的方法，和一些基础的训练信息写到yaml中，给inference用
        :return:
        """


"""Usage
model = Basetrainer(model='resnet152',
                    train_path='E:/Dataset_log/leaf_test/train/',
                    val_path='E:/Dataset_log/leaf_test/valid/',
                    weight_path='',
                    image_size=224,
                    save_path='E:/Train_log/Drone_thesis/Classification/ResNet152/exp4_codecheck/',
                    batch_size=10,
                    num_class=23,
                    device='cuda')
model.train(num_epochs=150)
"""

def main():
    model = CustomTrainer(cfg='../configs/sample.yaml')


if __name__ == '__main__':
    main()
