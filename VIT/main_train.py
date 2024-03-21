# 导入所需库
import os.path
from multiprocessing.spawn import freeze_support
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# 导入自建类
from data_preprocessing import PlantDataModule
from models import *
from CONFIG import CONFIG

if __name__ == '__main__':
    freeze_support()
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    # 创建数据模块实例
    datamodule = PlantDataModule(data_dir="../plant_dataset", CONFIG=CONFIG)
    datamodule.setup(stage="train")

    # 创建ViT模型实例
    lit_model = ViTBase16(cfg=CONFIG)

    # 创建TensorBoardLogger实例
    tb_logger = TensorBoardLogger("./tb_logs", name="plants")

    # 创建Trainer实例
    trainer = Trainer(
        max_epochs=CONFIG.num_epochs,
        accumulate_grad_batches=CONFIG.accum,
        precision=CONFIG.precision,
        callbacks=earlystopping,
        logger=tb_logger,
    )

    # 训练模型
    trainer.fit(lit_model, datamodule)

    # 保存训练误差曲线和验证误差曲线
    if not os.path.isdir("./tb_logs/logs"):
        os.makedirs("./tb_logs/logs")
    plt.plot(lit_model.train_loss_history)
    plt.plot(lit_model.val_loss_history)
    plt.title('Vision Transformer Model Loss with Dropout')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./tb_logs/logs/loss_curve.png')
    plt.close()

    # 保存训练F1曲线和验证F1曲线
    plt.plot(lit_model.train_f1_history)
    plt.plot(lit_model.val_f1_history)
    plt.title('Vision Transformer Model F1 scores with Dropout')
    plt.ylabel('F1 scores')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./tb_logs/logs/f1_curve.png')
    plt.close()

    # 保存训练准确率曲线和验证准确率曲线
    plt.plot(lit_model.train_acc_history)
    plt.plot(lit_model.val_acc_history)
    plt.title('Vision Transformer Model Accuracy with Dropout')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./tb_logs/logs/accuracy_curve.png')
    plt.close()

