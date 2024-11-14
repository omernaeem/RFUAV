from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import os
import yaml
from utils.build import build_from_cfg, check_cfg
from utils.logger import colorful_logger
import cv2
from abc import abstractmethod
from .metrics.base_metric import EVAMetric
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
METRIC = os.path.join(current_dir, './metrics')
sys.path.append(METRIC)


class Basetrainer:
    """Usage
    model = Basetrainer(model(str) = 'resnet152',
                        train_path(str) = '',
                        val_path(str) = '',
                        weight_path(str) = '',
                        image_size(int) = 224,
                        save_path(str) = '',
                        batch_size(int) = 32,
                        num_class(int) = 23,
                        device(str) = 'cuda'/'cpu',
                        shuffle(bool) = False,
                        log_file(str) = ''
                        lr(float) = 0.0001)
    model.train(num_epochs=50)
    the network we support have:
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
    "swin_v2_t", "swin_v2_s", "swin_v2_b", "mobilenet_v3_small"
    "mobilenet_v3_large", provided by torch.nn
    """
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
                 lr: float = 0.0001
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
        self.lr = lr
        self.logger = self.set_logger(os.path.join(save_path, log_file))
        self.criterion = criterion  # initializing the loss function
        self.set_up(model=model, train_path=train_path, val_path=val_path,
                    pretrained=pretrained, weight_path=weight_path)

    def set_up(self, train_path, val_path, pretrained, weight_path, model='resnet18'):

        self.logger.log_with_color(f"Loading model: {model}")

        if os.path.exists(weight_path):
            pretrained = False

        if not os.path.exists(pretrained):
            self.logger.log_with_color("Pretrained model not found, using default weight")
            pretrained = True

        self.model = model_init_(model_name=model, num_class=self.num_class, pretrained=pretrained)

        if os.path.exists(weight_path):
            self.load_pretrained_weights(weight_path)
            self.logger.log_with_color(f"Loading pretrained weights from: {weight_path}")

        self.model.to(self.device)
        self.logger.log_with_color(f"{model} loaded onto device: {self.device}")

        # initializing the dataset
        self.logger.log_with_color(f"Loading dataset from: {train_path} and {val_path}")
        _train_set = datasets.ImageFolder(root=train_path, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]))

        self.train_set = DataLoader(_train_set, batch_size=self.batch_size, shuffle=self.shuffle)

        _val_set = datasets.ImageFolder(root=val_path, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]))
        self.val_set = DataLoader(_val_set, batch_size=self.batch_size, shuffle=self.shuffle)

        # initializing optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    @abstractmethod
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.logger.log_with_color(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in self.train_set:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # backward
                loss.backward()
                self.optimizer.step()

                # acc & loss
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            train_loss = running_loss / len(self.train_set)
            train_acc = 100 * correct / total
            self.logger.log_with_color(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
            metrics = self.val
            self.logger.log_with_color(f'Validation Loss: {metrics["loss"]:.4f}, Validation Accuracy: {metrics["acc"]:.2f}%')
            self.save_model(metrics['acc'], epoch)

    @property
    def val(self):
        self.logger.log_with_color("Starting validation...")
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probabilities = []
        val_total_labels = []
        with torch.no_grad():
            for val_images, val_labels in self.val_set:
                val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                val_outputs = self.model(val_images)
                for val_output in val_outputs:
                    val_probabilities.append(list(torch.softmax(val_output, dim=0)))
                val_loss += self.criterion(val_outputs, val_labels).item()
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()
                val_total_labels.append(val_labels)
        _val_total_labels = torch.concat(val_total_labels, dim=0)
        _val_probabilities = torch.tensor(val_probabilities)
        metrics = EVAMetric(preds=_val_probabilities.to(self.device),
                            labels=_val_total_labels,
                            num_classes=self.num_class,
                            tasks=('f1', 'precision'),
                            topk=(1, 3, 5),
                            save_path=self.save_path,
                            classes_name=self.train_set.dataset.classes)

        metrics['acc'] = 100 * val_correct / val_total
        metrics['total_loss'] = val_loss / len(self.val_set)
        return metrics

    def save_model(self, val_acc, epoch):
        """
        Save the model after each epoch and track the best model based on validation accuracy.
        """
        checkpoint_path = os.path.join(self.save_path, f'{self.model._get_name()}_epoch_{epoch + 1}.pth')
        self.logger.log_with_color(f'Model saved at {checkpoint_path} (Validation Accuracy: {val_acc:.2f}%)')
        torch.save(self.model.state_dict(), checkpoint_path)

        # Save the best model if current validation accuracy is higher than the best recorded one
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_model = self.model.state_dict()
            best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(self.best_model, best_model_path)
            self.logger.log_with_color(f'New best model saved with Accuracy: {val_acc:.2f}%')

    def set_logger(self, log_file):

        logger = colorful_logger(name='Train', logfile=log_file)
        return logger

    def load_pretrained_weights(self, weight_path: str):

        if os.path.exists(weight_path):
            self.logger.log_with_color(f"Loading pretrained weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.log_with_color(f"Successfully loaded pretrained weights from: {weight_path}")
        else:
            self.logger.log_with_color(f"Pretrained weights file not found at: {weight_path}. Skipping weight loading.")


def model_init_(model_name, num_class, pretrained=True):

    # resnet series model
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

    # ViT series model
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

    # SiwnTrans series model
    elif model_name == "swin_v2_t":
        model = models.swin_v2_t(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)
    elif model_name == "swin_v2_s":
        model = models.swin_v2_s(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)
    elif model_name == "swin_v2_b":
        model = models.swin_v2_b(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)

    # Mobilenet series model
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
            super().__init__(
                model=self.parameters['model'],
                train_path=self.parameters['train'],
                val_path=self.parameters['val'],
                num_class=self.parameters['num_classes'],
                save_path=self.parameters['save_path'],
                weight_path=self.parameters['weights'],
                device=self.parameters['device'],
                batch_size=self.parameters['batch_size'],
                shuffle=self.parameters['shuffle'],
                image_size=self.parameters['image_size'],
                lr=self.parameters['lr'],
            )
        else:
            super().__init__(Basetrainer)

        self.class_idx = self.train_set.dataset.class_to_idx
        if self.save_yaml:
            self.logger.log_with_color(f"Saving yaml file at {self.parameters['save_path']}")

    @property
    def save_yaml(self):

        self.parameters['class_names'] = self.class_idx
        with open(os.path.join(self.save_path, 'config.yaml'), 'w', encoding='utf-8') as file:
            yaml.dump(self.parameters, file, allow_unicode=True)
        return True

    @property
    def train(self):
        num_epochs = self.parameters['num_epochs']

        for epoch in range(num_epochs):
            self.logger.log_with_color(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in self.train_set:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward propagation
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # backward propagation
                loss.backward()
                self.optimizer.step()

                # acc & loss
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            train_loss = running_loss / len(self.train_set)
            train_acc = 100 * correct / total
            self.logger.log_with_color(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
            metrics = self.val
            self.logger.log_with_color(f'Validation Loss: {metrics["loss"]:.4f},')
            self.logger.log_with_color(f' Validation Accuracy: {metrics["acc"]:.2f}%,')
            self.logger.log_with_color(f' Validation macro_F1: {metrics["f1"]["macro_f1"]}')
            self.logger.log_with_color(f' Validation micro_F1: {metrics["f1"]["micro_f1"]}')
            self.logger.log_with_color(f' Validation mAP: {metrics["mAP"]["mAP"]}')
            self.logger.log_with_color(f' Validation Top-k Accuracy: {metrics["Top-k"]}')

            self.save_model(metrics, epoch)


# for test--------------------------------------------------------------------------------------------------------------
def show_img_in_dataloader(images):
    """Imshow for Tensor."""
    images = images.numpy().transpose((1, 2, 0))
    cv2.imshow('test', images)
    cv2.waitKey(0)
