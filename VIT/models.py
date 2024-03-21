import numpy as np
import pytorch_lightning as pl
import timm
import torch
from matplotlib import pyplot as plt
from torchmetrics import F1Score, ConfusionMatrix
from torchmetrics import Accuracy, Precision, Recall
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score
from CONFIG import CONFIG


# 定义ViT模型
class ViTBase16(pl.LightningModule):
    def __init__(self, cfg=CONFIG, pretrained=True):
        super(ViTBase16, self).__init__()

        # 创建ViT模型和相关组件
        self.CONFIG = cfg
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, self.CONFIG.num_classes)
        self.f1 = F1Score(num_labels=self.CONFIG.num_classes, task="multilabel")
        self.cm = ConfusionMatrix(num_labels=self.CONFIG.num_classes, task="multilabel")
        self.accuracy = Accuracy(num_labels=self.CONFIG.num_classes, task="multilabel")
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.lr = self.CONFIG.lr

        self.train_step_loss = []
        self.train_step_f1 = []
        self.train_step_acc = []
        self.val_step_loss = []
        self.val_step_f1 = []
        self.val_step_acc = []
        self.train_loss_history = []
        self.train_f1_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_f1_history = []
        self.val_acc_history = []

    def forward(self, x, *args, **kwargs):
        # 前向传播
        return self.model(x)

    def configure_optimizers(self):
        # 配置优化器和学习率调度器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.CONFIG.t_max,
                                                                    eta_min=self.CONFIG.min_lr)
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def on_train_epoch_end(self):
        # 记录整个epoch的平均训练损失和F1分数
        avg_loss = torch.stack(self.train_step_loss).mean()
        self.train_loss_history.append(avg_loss.item())
        self.train_step_loss.clear()
        avg_f1 = torch.stack(self.train_step_f1).mean()
        self.train_f1_history.append(avg_f1.item())
        self.train_step_f1.clear()
        avg_acc = torch.stack(self.train_step_acc).mean()
        self.train_acc_history.append(avg_acc.item())
        self.train_step_acc.clear()

    def on_validation_epoch_end(self):
        # 记录整个epoch的平均训练损失和F1分数
        avg_loss = torch.stack(self.val_step_loss).mean()
        self.val_loss_history.append(avg_loss.item())
        self.val_step_loss.clear()
        avg_f1 = torch.stack(self.val_step_f1).mean()
        self.val_f1_history.append(avg_f1.item())
        self.val_step_f1.clear()
        avg_acc = torch.stack(self.val_step_acc).mean()
        self.val_acc_history.append(avg_acc.item())
        self.val_step_acc.clear()

    def training_step(self, batch, batch_idx):
        # 训练步骤
        image = batch['images']
        target = batch['target']
        output = self.model(image)
        loss = self.criterion(output, target)
        score = self.f1(self.sigmoid(output), target.clone().detach().to(torch.int32))
        # 计算准确率
        acc = accuracy_score(target.cpu(), (self.sigmoid(output) > 0.5).cpu())
        logs = {'train_loss': loss, 'train_acc': acc, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # 记录训练误差和F1分数
        self.train_step_loss.append(loss)
        self.train_step_f1.append(score)
        self.train_step_acc.append(torch.tensor(acc))
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证步骤
        image = batch['images']
        target = batch['target']
        output = self.model(image)
        loss = self.criterion(output, target)
        score = self.f1(self.sigmoid(output), target.clone().detach().to(torch.int32))
        # 计算准确率
        acc = accuracy_score(target.cpu(), (self.sigmoid(output) > 0.5).cpu())
        logs = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # 记录验证误差和F1分数
        self.val_step_loss.append(loss)
        self.val_step_f1.append(score)
        self.val_step_acc.append(torch.tensor(acc))
        return loss
