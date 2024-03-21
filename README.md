# Identify-the-category-of-foliar-diseases-in-apple-trees

<p align="center">
  <a href="./README_en.md">English</a> |
  <a href="./README.md">简体中文</a>
</p>


本项目探讨了Kaggle举办的FGVC8(CVPR2021-workshop)Plant Pathology-2021数据集
分类任务。

通过将标签进行独热编码转化为多标签分类问题，我们研究了两种卷积神经网络模型来解决本分类任务。

我们分析了Plant Pathology-2021数据集的样本细节，并详细介绍了LeNet模型与Vision Transformer模型的构建、训练与优化过程以及它们在Plant Pathology-2021分类任务上不同的性能差距。

我们还讨论了ArcFaceLoss与CrossEntropyLoss在本任务上的效果对比，以及对比数据集裁剪的效果。

_详情请见<a href="./template_Article.pdf">论文</a>_

## 特性

- **使用LeNet模型**: 可以达到65%准确率。
- **使用VIT模型**: 可达到60%准确率。
- **对比了ArcFaceLoss与CrossEntropyLoss的效果**: 详情请见<a href="./template_Article.pdf">论文</a>。

## 安装

要运行此项目，您需要安装 Python 3.8 或更高版本。首先克隆此仓库：

```bash
git clone https://github.com/LinYujupiter/Identify-the-category-of-foliar-diseases-in-apple-trees.git
cd Identify-the-category-of-foliar-diseases-in-apple-trees
```

## 环境配置

### 使用 Conda

如果您使用 Conda，可以按照以下步骤设置和激活虚拟环境：

1. **创建虚拟环境**：

   ```bash
   conda create -n Identify-the-category-of-foliar-diseases-in-apple-trees python=3.8
   ```

2. **激活虚拟环境**：

   ```bash
   conda activate Identify-the-category-of-foliar-diseases-in-apple-trees
   ```

3. **安装依赖**：

   在激活的虚拟环境中，运行：

   ```bash
   pip install -r requirements.txt
   ```

### 不使用 Conda

如果您不使用 Conda，可以直接使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

## 运行

在安装了所有依赖之后，您可以通过以下命令启动VIT模型训练：

```bash
cd VIT
python3 main_train.py
```

您可以通过以下命令启动LeNet模型训练：

```bash
cd LeNet
python3 main_train.py
```


## 使用

训练完成后，您可以通过以下命令启动测试：

```bash
python3 main_test.py
```

## 开发

- **pytorch**: 用于训练VIT模型。
- **tensorflow**: 用于训练LeNet模型。

## 贡献

我们欢迎任何形式的贡献，无论是新功能的提议、代码改进还是问题报告。请确保遵循最佳实践和代码风格指南。
