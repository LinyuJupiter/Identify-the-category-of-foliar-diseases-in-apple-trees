from models import *
import torchviz
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin/'
# 创建一个 ViTBase16 模型实例
model = ViTBase16()

# 生成一个随机输入张量，用于生成计算图
dummy_input = torch.randn((1, 3, 224, 224))

# 可视化计算图
torchviz.make_dot(model(dummy_input), params=dict(model.named_parameters())).render("ViTBase16", format="png")