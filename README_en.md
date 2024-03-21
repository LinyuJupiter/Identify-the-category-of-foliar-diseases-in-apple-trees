# Identify-the-category-of-foliar-diseases-in-apple-trees

<p align="center">
  <a href="./README_en.md">English</a> |
  <a href="./README.md">简体中文</a>
</p>


This project explores the classification task on the Plant Pathology-2021 dataset hosted by Kaggle for FGVC8 (CVPR2021-workshop).

By converting the labels into one-hot encoding to transform it into a multi-label classification problem, we investigated two convolutional neural network models to tackle this classification task.

We analyzed the sample details of the Plant Pathology-2021 dataset and provided a detailed overview of the construction, training, and optimization processes of the LeNet model and Vision Transformer (VIT) model, as well as their different performance gaps in the Plant Pathology-2021 classification task.

We also discussed the comparison between ArcFaceLoss and CrossEntropyLoss on this task, as well as the effect of dataset cropping.

_For more details, please refer to the <a href="./template_Article.pdf">paper</a>._

## Features

- **Using LeNet Model**: Achieves an accuracy of 65%.
- **Using VIT Model**: Attains an accuracy of 60%.
- **Comparison of ArcFaceLoss and CrossEntropyLoss**: For details, please see the <a href="./template_Article.pdf">paper</a>.

## Installation

To run this project, you need to install Python 3.8 or higher. First, clone this repository:

```bash
git clone https://github.com/LinYujupiter/Identify-the-category-of-foliar-diseases-in-apple-trees.git
cd Identify-the-category-of-foliar-diseases-in-apple-trees
```

## Environment Setup

### Using Conda

If you're using Conda, you can set up and activate a virtual environment as follows:

1. **Create a virtual environment**:

   ```bash
   conda create -n Identify-the-category-of-foliar-diseases-in-apple-trees python=3.8
   ```

2. **Activate the virtual environment**:

   ```bash
   conda activate Identify-the-category-of-foliar-diseases-in-apple-trees
   ```

3. **Install dependencies**:

   Within the activated virtual environment, run:

   ```bash
   pip install -r requirements.txt
   ```

### Without Conda

If you're not using Conda, you can directly install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Running

After installing all dependencies, you can start training the VIT model with the following command:

```bash
cd VIT
python3 main_train.py
```

You can start training the LeNet model with the following command:

```bash
cd LeNet
python3 main_train.py
```

## Usage

After training, you can start testing with the following command:

```bash
python3 main_test.py
```

## Development

- **PyTorch**: Used for training the VIT model.
- **TensorFlow**: Used for training the LeNet model.

## Contribution

We welcome contributions in any form, whether it's proposing new features, improving code, or reporting issues. Please make sure to follow best practices and code style guidelines.