from models.example_model import *
import os
from PIL import Image
import scipy.io as scio
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.layers import advanced_activations

data_dir1 = "D:\PycharmProject\化肥论文\data"
data_dir2 = "D:\PycharmProject\化肥论文/test"
a = np.zeros([256, 256])
b = np.expand_dims(a, axis=-1)

def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = (np.array(image) / 255.0)
        data1 = np.concatenate((b, data), axis=2)
        label = int(fname.split("_")[0])
        datas.append(data1)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)


    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels

fpaths, datas, labels = read_data(data_dir1)
fpaths2, datas2, labels2 = read_data(data_dir2)
# datas = np.concatenate((f, datas), axis=3)

# 计算有多少类图片
num_classes = len(set(labels))
labels = np_utils.to_categorical(labels, 2)
print(labels.shape)
# 验证集
x_dev = datas[91:110, :]
y_dev = labels[91:110, :]
# 测试集
x_test = datas2
labels2 = np_utils.to_categorical(labels2, 2)
print(x_test.shape)
print(labels2)

input_shape = (256, 256, 4)

# dingliang
model = Sequential()

model.add(QuaternionConv2D(8, 1, strides=3, padding="same", input_shape=input_shape))
normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9,
                                 weights=None, beta_init='zero', gamma_init='one')
advanced_activations.LeakyReLU(alpha=0.3)
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='valid'))

model.add(QuaternionConv2D(16, 3, strides=3, padding="same", input_shape=input_shape))
normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9,
                                 weights=None, beta_init='zero', gamma_init='one')
advanced_activations.LeakyReLU(alpha=0.3)
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='valid'))

model.add(QuaternionConv2D(32, 3, strides=3, padding="same", input_shape=input_shape))
normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9,
                                 weights=None, beta_init='zero', gamma_init='one')
advanced_activations.LeakyReLU(alpha=0.3)
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='valid'))

model.add(QuaternionConv2D(64, 5, strides=3, padding="same", input_shape=input_shape))
normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9,
                                 weights=None, beta_init='zero', gamma_init='one')
advanced_activations.LeakyReLU(alpha=0.3)
model.add(Activation('relu'))
# model.add(MaxPooling2D(2, padding='valid'))

model.add(Flatten())

model.add(Dense(256))
normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9,
                                 weights=None, beta_init='zero', gamma_init='one')
#model.add(Activation('tanh'))
model.add(Activation('relu'))
# model.add(Dropout(0.3))

model.add(Dense(2))

# print model 输出模型结构和参数数目（打印出模型概述信息）
model.summary()
#complile 编译 交叉熵损失函数 Adam优化算法 以正确率作为指标
model.compile(loss='mean_squared_error', optimizer='Adam')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
#train 训练 批处理样本数目，训练epoch数，交叉验证数据比例，训练过程的输出
history=model.fit(datas, labels, validation_data=(x_dev, y_dev), epochs=500, batch_size=10,
          verbose=1, shuffle=False, callbacks=[early_stopping])#,validation_data=(X_test,Y_test))

# loss曲线
#plt.grid(True)  # 画网格
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
# model.save('model1.h5') 保存训练好的模型

#test 在测试集上测试训练正确率
yhat = model.predict(x_test)
y = model.predict(datas)
print(yhat)
#print(x_test.shape)

rmse = sqrt(mean_squared_error(labels2, yhat))

r2 = r2_score(labels2,yhat)

print('Test RMSEP: %.3f' % rmse)
print('The r2:%.3f' % r2)
#print(y_test)
#print(yhat)

#plt.scatter(y_test,yhat)
#plt.show()
'''
rmsec = sqrt(mean_squared_error(y_train, y))

r2c = r2_score(y_train,y)

print('Test RMSEP: %.3f' % rmsec)
print('The r2:%.3f' % r2c)
'''