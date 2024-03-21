import numpy as np
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torchmetrics import Accuracy, Precision, Recall
import torch
from data_preprocessing import get_df
from ArcLoss import ArcLoss
from CONFIG import *

# 加载保存好的模型
loaded_model = load_model(f'./models/{ArcLoss_OR_CrossEntropyLoss}_model.keras')

X_test, y_test = get_df("test")
test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test)

# 模型预测
y_pred = loaded_model.predict(X_test)

# 将概率转换为二进制预测
threshold = 0.5  # 可根据需要调整阈值
y_pred_binary = (y_pred > threshold).astype(int)

# 计算查准率和查全率
test_accuracy = accuracy_score(y_test, y_pred_binary)
test_precision = precision_score(y_test, y_pred_binary, average='weighted')
test_recall = recall_score(y_test, y_pred_binary, average='weighted')
# 另一种计算查准率和查全率
# accuracy = Accuracy(num_labels=6, task="multilabel")
# precision = Precision(num_labels=6, task="multilabel")
# recall = Recall(num_labels=6, task="multilabel")
# test_accuracy = accuracy(torch.tensor(y_pred_binary), torch.tensor(y_test))
# test_precision = precision(torch.tensor(y_pred_binary), torch.tensor(y_test))
# test_recall = recall(torch.tensor(y_pred_binary), torch.tensor(y_test))
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Precision: {test_precision:.4f}')
print(f'Recall: {test_recall:.4f}')
