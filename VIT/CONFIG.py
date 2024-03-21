import torch


# 定义配置类
class CONFIG:
    model_name = 'vit_base_patch16_224'
    pretrained = True
    img_size = 224
    num_classes = 6
    lr = 5e-4
    min_lr = 1e-6
    t_max = 20
    num_epochs = 50
    batch_size = 16
    accum = 1
    precision = "16-mixed"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
