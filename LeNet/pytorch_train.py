import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torchvision import transforms
from keras.preprocessing import image
from sklearn import preprocessing
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置路径
path = '../plant_dataset/'
# 设置超参数
TARGET_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.5  # 初始分类阈值
num = 1  # 第一次训练
save_model = False  # 是否保存模型


def get_df(dir_path, mode):
    # 处理数据
    # 读取训练数据
    df = pd.read_csv(os.path.join(dir_path + f'{mode}/{mode}_label.csv'))
    # 对目标变量进行独热编码
    train = df.copy()
    train['labels'] = df['labels'].apply(lambda string: string.split(' '))
    s = list(train['labels'])
    mlb = preprocessing.MultiLabelBinarizer()
    trainx = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=train.index)
    trainx['images'] = train['images']
    return trainx


# 定义自定义数据集
class PlantDataset(Dataset):
    def __init__(self, dataframe, directory, mode="train", transform=None):
        self.dataframe = dataframe
        self.directory = os.path.join(directory, f"{mode}_256/images")
        self.transform = transform
        # 提取标签
        self.y = np.array(dataframe.drop(['images'], axis=1))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        idx = int(idx)  # 将idx转换为整数
        img_name = os.path.join(self.directory, self.dataframe["images"][idx])
        image = self.load_image(img_name)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(TARGET_SIZE, TARGET_SIZE))
        img = image.img_to_array(img)
        # img = img / 255.0  # 将像素值标准化在0到1之间
        return img


# 定义数据转换
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 创建数据集实例
train_dataset = PlantDataset(get_df(path, "train"), path, "train", transform=data_transform)
val_dataset = PlantDataset(get_df(path, "val"), path, "val", transform=data_transform)
test_dataset = PlantDataset(get_df(path, "test"), path, "test", transform=data_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def calculate_f1_score(y_true, y_pred, threshold):
    y_pred_binary = (y_pred > threshold).float()
    true_positives = torch.sum((y_pred_binary * y_true).float(), dim=0)
    false_positives = torch.sum(((1 - y_true) * y_pred_binary).float(), dim=0)
    false_negatives = torch.sum((y_true * (1 - y_pred_binary)).float(), dim=0)

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1.mean()


# 实例化LeNet模型
model = LeNet().to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_losses = []
val_losses = []
test_losses = []

train_accuracies = []
val_accuracies = []
test_accuracies = []

train_f1_scores = []
val_f1_scores = []
test_f1_scores = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    train_f1 = []

    for batch in train_loader:
        inputs, labels = batch['image'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # 对每个输出应用 sigmoid 函数
        predicted = torch.sigmoid(outputs)
        # 根据阈值（例如0.5）确定每个类别的预测结果
        predicted_binary = (predicted > threshold).float()
        total_train += labels.size(0)
        correct_train += ((predicted_binary == labels).sum(dim=1) == labels.size(1)).sum().item()

        # 计算 F1 分数
        train_f1_batch = calculate_f1_score(labels, predicted, threshold)
        train_f1.append(train_f1_batch.item())

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    avg_train_f1 = np.mean(train_f1)
    train_f1_scores.append(avg_train_f1)

    # 模型评估
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    avg_val_f1 = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # 计算验证准确率
            predicted = torch.sigmoid(outputs)
            predicted_binary = (predicted > threshold).float()
            total_val += labels.size(0)
            correct_val += ((predicted_binary == labels).sum(dim=1) == labels.size(1)).sum().item()
            # 计算F1分数
            val_f1 = calculate_f1_score(labels, torch.sigmoid(outputs), threshold)
            avg_val_f1.append(val_f1)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    avg_val_f1_scores = torch.stack(avg_val_f1).mean(dim=0)
    best_val_f1_score = avg_val_f1_scores.cpu().item()
    val_f1_scores.append(best_val_f1_score)

    # 测试集评估
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    avg_test_f1 = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # 计算测试准确率
            predicted = torch.sigmoid(outputs)
            predicted_binary = (predicted > threshold).float()
            total_test += labels.size(0)
            correct_test += ((predicted_binary == labels).sum(dim=1) == labels.size(1)).sum().item()

            # 计算 F1 分数
            test_f1 = calculate_f1_score(labels, torch.sigmoid(outputs), threshold)
            avg_test_f1.append(test_f1)
    avg_test_f1_scores = torch.stack(avg_test_f1).mean(dim=0).cpu().item()
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct_test / total_test
    test_losses.append(avg_test_loss)
    test_f1_scores.append(avg_test_f1_scores)
    if save_model:
        if not os.path.isdir(f'./plant/{num}'):
            os.makedirs(f'./plant/{num}')
        torch.save(model.state_dict(), f'./plant/{num}/lenet_model_epoch_{epoch + 1}.pth')
    # 打印训练和测试信息
    print(f'Epoch [{epoch + 1}/{EPOCHS}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4%}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4%}, '
          f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4%}')

# 绘制训练和测试的准确度曲线
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('LeNet Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'pytorch_accuracy_plot.png')
plt.close()  # 关闭绘图，以防止与下一个图表叠加

# 绘制训练和测试的损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('LeNet Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'pytorch_loss_plot.png')
plt.close()  # 关闭绘图，以防止与下一个图表叠加

# 绘制验证和测试的F1曲线
plt.plot(train_f1_scores, label='Val F1 scores')
plt.plot(val_f1_scores, label='Val F1 scores')
plt.plot(test_f1_scores, label='Test F1 scores')
plt.title('LeNet Model F1 scores')
plt.xlabel('Epoch')
plt.ylabel('F1 scores')
plt.legend()
plt.savefig(f'pytorch_F1 score_plot.png')
plt.close()  # 关闭绘图，以防止与下一个图表叠加
