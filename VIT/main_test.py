# 导入所需库
from multiprocessing.spawn import freeze_support
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch import sigmoid

# 导入自建类
from data_preprocessing import PlantDataModule
from models import *
from CONFIG import CONFIG

if __name__ == '__main__':
    freeze_support()
    PATH_TO_CHECKPOINT = "./tb_logs/plants/version_12/checkpoints/epoch=17-step=3366.ckpt"
    # 加载训练好的模型权重
    trained_model = ViTBase16.load_from_checkpoint(PATH_TO_CHECKPOINT).to(CONFIG.device)

    # 创建测试数据模块
    test_datamodule = PlantDataModule(data_dir="../plant_dataset", CONFIG=CONFIG)
    test_datamodule.setup(stage="test")

    # 设置模型为评估模式
    trained_model.eval()
    trained_model.to(CONFIG.device)

    # 初始化列表以保存预测和真实标签
    all_acc = []
    all_precision = []
    all_recall = []
    all_probs = []  # 保存预测概率的列表

    accuracy = Accuracy(num_labels=CONFIG.num_classes, task="multilabel").to(CONFIG.device)
    precision = Precision(num_labels=CONFIG.num_classes, task="multilabel").to(CONFIG.device)
    recall = Recall(num_labels=CONFIG.num_classes, task="multilabel").to(CONFIG.device)
    f1 = F1Score(num_labels=CONFIG.num_classes, task="multilabel").to(CONFIG.device)

    with torch.no_grad():
        for batch in test_datamodule.test_dataloader():
            images = batch['images'].to(CONFIG.device)
            targets = batch['target'].to(CONFIG.device)

            # 前向传播
            output = trained_model(images)
            acc = accuracy_score(targets.cpu(), (sigmoid(output) > 0.5).cpu())
            all_acc.append(torch.tensor(acc))
            pre = precision(sigmoid(output), targets)
            all_precision.append(pre)
            rec = recall(sigmoid(output), targets)
            all_recall.append(rec)

            # 获取预测概率
            probabilities = torch.sigmoid(output)  # 如果使用sigmoid激活函数
            all_probs.append(probabilities.cpu().numpy())

    # 计算平均值
    avg_acc = torch.mean(torch.tensor(all_acc))
    avg_precision = torch.mean(torch.tensor(all_precision))
    avg_recall = torch.mean(torch.tensor(all_recall))
    print(f"Accuracy: {avg_acc:.4%}")
    print(f"Precision: {avg_precision:.4%}")
    print(f"Recall: {avg_recall:.4%}")

    # 将预测概率和真实标签转换为一维数组
    all_probs = np.concatenate(all_probs)
    all_targets = test_datamodule.test_dataset.labels

    # 计算PR曲线
    precision_curve, recall_curve, _ = precision_recall_curve(all_targets.ravel(), all_probs.ravel())

    # 计算AUC值
    auc_value = auc(recall_curve, precision_curve)

    # 绘制PR曲线
    plt.figure(figsize=(8, 8))
    plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {auc_value:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
