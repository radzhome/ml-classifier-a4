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


def dataset(filename):
    """
    Splits dataset into train and test
    :param filename: str, name of file i.e. A4train or A4val
    :return: tuple(train_x, train_y, test_x, test_y)
    """
    file = f"a4data/{filename}.csv"
    x = np.loadtxt(file, delimiter=",")
    y = x[:, 0]
    x = x[:, 1:]

    train_x = x[: round(len(x) * 0.80)]  # 80%
    train_y = y[: round(len(x) * 0.80)]  # 80%

    test_x = x[round(len(x) * 0.80):]  # 20%
    test_y = y[round(len(x) * 0.80):]  # 20%

    return train_x, train_y, test_x, test_y


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


def main():
    train_x, train_y, test_x, test_y = dataset('A4train')
    vector = train_x[1]  # change element to get different pics
    plotImg(vector)


if __name__ == '__main__':
    main()
