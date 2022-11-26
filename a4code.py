import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate
from keras.utils import to_categorical


def plotImg(x):
    """
    You can use the
    following function to visualize an image:
    :param x:
    :return:
    """
    img = x.reshape((84, 28))
    plt.imshow(img, cmap='gray')
    plt.show()