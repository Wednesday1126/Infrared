import numpy as np
import matplotlib.pyplot as plt
# from pyts.image import GASF, GADF
from pyts.image import GramianAngularField

x = np.loadtxt(open("3.csv", "rb"), delimiter=",", skiprows=0)
print(x.shape)

# 取要做成图像的一行（一个样本）
image_size = 256
number = 100
for i in range(number):
    X = x[i+99, :].reshape(1, -1)
    print(type(X), X.shape, i)
    gadf = GramianAngularField(image_size=image_size, method='difference')
    # gasf = GramianAngularField(image_size=image_size, method='summation')
    # gadf = GADF(image_size)
    X_gadf = gadf.fit_transform(X)
    plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
    plt.axis('off')
    plt.savefig('D:/PycharmProject/化肥论文/you/wu{}.jpeg'.format(i), bbox_inches='tight', pad_inches=0)
    # plt.clf()
    # plt.show()
