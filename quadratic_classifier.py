'''
Data classification using quadratic decision boundary
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from visualization import visualize_data_2
from common import linear_classifier


def calc_square(X):
    '''
    :param X: numpy array, data
    :return: numpy array, data with squared values
    '''

    m = X.shape[0]
    Y = np.zeros((m, int(n*(n+1)/2)))
    k = 0
    for i in range(n):
        for j in range(i, n):
            Y[:,k] = X[:,i] * X[:,j]
            k += 1
    return Y


if __name__ == '__main__':

    # Read data - 2 calsses
    X1 = np.array(pd.read_csv('./data/t3.2_class1.csv', names=['feature1', 'feature2']))
    X2 = np.array(pd.read_csv('./data/t3.2_class2.csv', names=['feature1', 'feature2']))

    # Display data
    visualize_data_2(X1, X2)
    plt.title('Data')
    plt.show()
    n = X1.shape[1]
    m = X1.shape[0]

    # Calculate squared features
    Y1 = calc_square(X1)
    Y2 = calc_square(X2)
    Z1 = np.concatenate((Y1, X1), axis=1)
    Z2 = np.concatenate((Y2, X2), axis=1)

    # Implement linear classifier for features and for squared features
    V, v0, acc = linear_classifier(Z1, Z2)

    # Get parameters
    a = V[:int(n*(n+1)/2)]
    v = V[int(n*(n+1)/2):]

    # Plot decision boundary
    x_plot = np.linspace(min(X1[:,0].min(), X2[:,0].min()), max(X1[:,0].max(), X2[:,0].max()), 500)

    # Calculate y_plot=x2=(-b+-sqrt(b^2 - 4dc))/(2d)
    d = a[2]
    b = a[1] * x_plot + v[1]
    c = v[0] * x_plot + a[0] * np.square(x_plot) + v0
    y1_plot = (-b + np.sqrt(np.square(b) - 4 * d * c)) / (2 * d)
    y2_plot = (-b - np.sqrt(np.square(b) - 4 * d * c)) / (2 * d)

    x_plot = np.concatenate((x_plot, x_plot))
    y_plot = np.concatenate((y1_plot, y2_plot))

    # Display data and decision boundary
    visualize_data_2(X1, X2)
    plt.plot(x_plot, y_plot, '.g', label='Decision boundary')
    plt.legend()
    plt.show()

    print('Accuracy = {}%'.format(acc*100))
