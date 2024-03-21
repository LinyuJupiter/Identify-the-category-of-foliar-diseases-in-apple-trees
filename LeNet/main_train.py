import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from data_preprocessing import get_df
from ArcLoss import ArcLoss
from CONFIG import *

# 获取训练集和验证集数据
X_train, y_train = get_df("train")
X_val, y_val = get_df("val")

# 构建卷积神经网络模型
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(TARGET_SIZE, TARGET_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))
# 使用ArcLoss作为损失函数
if ArcLoss_OR_CrossEntropyLoss == "ArcLoss":
    loss = ArcLoss()
else:
    loss = 'binary_crossentropy'
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), batch_size=BATCH_SIZE)

# 保存整个模型
model.save(f'./models/{ArcLoss_OR_CrossEntropyLoss}_model_cut.keras')

# 绘制准确度图
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Deep LeNet Model Accuracy with Dropout')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'./logs/{ArcLoss_OR_CrossEntropyLoss}_accuracy_plot.png')
plt.close()  # 关闭绘图，以防止与下一个图表叠加

# 绘制损失图
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Deep LeNet Model Loss with Dropout')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'./logs/{ArcLoss_OR_CrossEntropyLoss}_loss_plot.png')
plt.close()  # 关闭绘图，以防止与下一个图表叠加
