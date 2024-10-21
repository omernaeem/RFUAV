from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import os
import logging

class trainer():
   def __init__(self,
                model:str,
                train_path:str,
                val_path:str,
                num_class:int,
                save_path:str,
                log_file:str="train.log",
                device:str="cuda",
                criterion=nn.CrossEntropyLoss(),
                batch_size:int=32,
                suffle:bool=False,
                ):
      
      self.batch_size = batch_size
      self.suffle = suffle
      self.num_class = num_class
      self.save_path = save_path
      self.best_acc = 0
      self.best_loss = 1e6
      self.best_epoch = 0
      self.best_model = None
      self.device = torch.device(device if torch.cuda.is_available() else "cpu")
      self.logger = self.set_logger(os.path.join(save_path, log_file))
      self.criterion = criterion  # initalizeing the loss function 
      self.set_up(model=model, train_path=train_path, val_path=val_path)
      
   def set_up(self, train_path, val_path, model='resnet18'):
      self.logger.info(f"Loading model: {model}")
      # resnet serise model
      if model == 'resnet18':
         self.logger.info(f"Using {model}")
         self.model = models.resnet18(pretrained=True)
         self.model.fc = nn.Linear(self.model.fc.in_features, self.num_class)
      elif model == "resnet34":
         self.logger.info(f"Using {model}")
         self.model = models.resnet34(pretrained=True)
         self.model.fc = nn.Linear(self.model.fc.in_features, self.num_class)
      elif model == 'resnet50':
         self.logger.info(f"Using {model}")
         self.model = models.resnet50(pretrained=True)
         self.model.fc = nn.Linear(self.model.fc.in_features, self.num_class)
      elif model == 'resnet101':
         self.logger.info(f"Using {model}")
         self.model = models.resnet101(pretrained=True)
         self.model.fc = nn.Linear(self.model.fc.in_features, self.num_class)
      elif model == 'resnet152':
         self.logger.info(f"Using {model}")
         self.model = models.resnet152(pretrained=True)
         self.model.fc = nn.Linear(self.model.fc.in_features, self.num_class)
         
      # ViT serise model
      elif model == "vit_b_16":
         self.logger.info(f"Using {model}")
         self.model = models.vit_b_16(pretrained=True)
         self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.num_class)
      elif model ==  "vit_b_32":
         self.logger.info(f"Using {model}")
         self.model = models.vit_b_32(pretrained=True)
         self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.num_class)
      elif model == "vit_l_16":
         self.logger.info(f"Using {model}")
         self.model = models.vit_l_16(pretrained=True)
         self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.num_class)
      elif model == "vit_l_32":
         self.logger.info(f"Using {model}")
         self.model = models.vit_l_32(pretrained=True)
         self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.num_class)
      elif model == "vit_h_14":
         self.logger.info(f"Using {model}")
         self.model = models.vit_h_14(pretrained=True)
         self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.num_class)

      # SiwnTrans serise mdoel
      elif model == "swin_v2_t":
         self.logger.info(f"Using {model}")
         self.model = models.swin_v2_t(pretrained=True)
         self.model.head = nn.Linear_layer(self.model.head.in_features, self.num_class)
      elif model == "swin_v2_s":
         self.logger.info(f"Using {model}")
         self.model = models.swin_v2_s(pretrained=True)
         self.model.head = nn.Linear_layer(self.model.head.in_features, self.num_class)
      elif model == "swin_v2_b":
         self.logger.info(f"Using {model}")
         self.model = models.swin_v2_b(pretrained=True)
         self.model.head = nn.Linear_layer(self.model.head.in_features, self.num_class)
         
      # Mobilnet serise model
      elif model == "mobilenet_v3_large":
         self.logger.info(f"Using {model}")
         self.model = models.mobilenet_v3_large(pretrained=True)
         self.model.classifier = nn.Linear_layer(self.model.classifier.in_features, self.num_class)
      elif model == "mobilenet_v3_small":
         self.logger.info(f"Using {model}")
         self.model = models.mobilenet_v3_small(pretrained=True)
         self.model.classifier = nn.Linear_layer(self.model.classifier.in_features, self.num_class)
         
      else:
         raise ValueError("model not supported")

      self.model.to(self.device)
      self.logger.info(f"{model} loaded onto device: {self.device}")

      # initializing the dataset
      self.logger.info(f"Loading dataset from: {train_path} and {val_path}")
      _train_set = datasets.ImageFolder(root=train_path, transform=transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]))
      self.train_set = DataLoader(_train_set, batch_size=self.batch_size, shuffle=self.suffle)
      
      _val_set = datasets.ImageFolder(root=val_path, transform=transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]))
      self.val_set = DataLoader(_val_set, batch_size=self.batch_size, shuffle=self.suffle)
      
      # initializing optimizer
      self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

   def train(self, num_epochs): 
      for epoch in range(num_epochs):
         self.logger.info(f"Epoch [{epoch+1}/{num_epochs}] started.")
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
         train_loss = running_loss/len(self.train_set)
         train_acc = 100 * correct / total
         self.logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
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
      print(f'Validation Loss: {val_loss/len(self.val_set)}, Validation Accuracy: {100 * val_correct / val_total}%')
      return 100*val_correct/val_total, val_loss/len(self.val_set)
      
   def save_model(self, val_acc, epoch):
      """
      Save the model after each epoch and track the best model based on validation accuracy.
      """
      checkpoint_path = os.path.join(self.save_path, f'{self.model._get_name()}_epoch_{epoch+1}.pth')
      self.logger.info(f'Model saved at {checkpoint_path} (Validation Accuracy: {val_acc:.2f}%)')
      torch.save(self.model.state_dict(), checkpoint_path)
      print(f'Model saved at {checkpoint_path}')
      
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
   
   """
   ToDo
   """
   def load_pretrained_weights(self, weight_path:str):
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



"""USAGE
model = trainer(model='resnet18',
              train_path='./data/train',
              val_path='./data/val',
              scheduler=None,
              device='cuda)
model.train(num__epoch)
"""
model = trainer(model='resnet18',
              train_path='/home/frank/Dataset/leaf_test/train',
              val_path='/home/frank/Dataset/leaf_test/valid',
              save_path='/home/frank/Train_res/exp1',
              num_class=10,
              device='cuda')
model.train(num_epochs=90)