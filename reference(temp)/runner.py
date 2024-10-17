from dataloader import SpectrumClassifyDataset


from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim


# CFG
train_path = "/home/frank/project/RFUAV/reference(temp)/dataset/train"
val_path = "/home/frank/project/RFUAV/reference(temp)/dataset/valid"



# 定义数据增强方法
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 需要输入224x224大小的图像
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的标准均值和方差
])

train_dataset = SpectrumClassifyDataset(root=train_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = SpectrumClassifyDataset(root=val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载预训练的 ResNet50 模型
resnet50 = models.resnet50(pretrained=True)

# 修改最后的全连接层，设置为你的类别数（假设有10个类别）
num_classes = 10
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于分类任务
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)  # 使用Adam优化器，学习率可以调整

# 将模型移动到GPU（如果有GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)
num_epochs = 10  # 训练的轮数

for epoch in range(num_epochs):
    resnet50.train()  # 将模型设置为训练模式
    
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 清除之前的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = resnet50(images)
        loss = criterion(outputs, labels)
        
        # 反向传播并优化
        loss.backward()
        optimizer.step()
        
        # 统计训练过程中的损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}%')

    # (可选) 每个 epoch 之后可以在验证集上测试模型性能
    resnet50.eval()  # 切换到评估模式
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = resnet50(val_images)
            val_loss += criterion(val_outputs, val_labels).item()
            _, val_predicted = val_outputs.max(1)
            val_total += val_labels.size(0)
            val_correct += val_predicted.eq(val_labels).sum().item()
    
    print(f'Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {100 * val_correct / val_total}%')
