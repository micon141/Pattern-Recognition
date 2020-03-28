""""
Desired output classification algorithm
Suppose that output is matrix G = [1 1 ... 1]
Calculate parameters W = (U U.T)⁻¹ U G where U is input features
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from visualization import visualize_data_2
from common import split_data


def plot_data(X1, X2, W):

    # Plot data and decision boundary
    X = np.concatenate((X1, X2))
    x_plot = np.linspace(min(X[:, 0]), max(X[:, 0]), num=50)
    y_plot = - W[1] / W[2] * x_plot - W[0] / W[2]
    visualize_data_2(X1, X2)
    plt.plot(x_plot, y_plot, '-g')
    plt.title('Data and decision boundary')
    plt.show()

def desired_output(X1, X2):
    '''
    :param X1: numpy array, features for class 1
    :param X2: numpy array, features for class 2
    '''

    # Add ones at the begining of the matrices X1 and X2
    Z1 = np.concatenate((np.ones((X1.shape[0], 1)), X1), axis=1)
    Z2 = np.concatenate((-np.ones((X2.shape[0], 1)), -X2), axis=1)
    # Input matrix
    U = np.concatenate((Z1, Z2)).T
    # Desired output
    G = np.ones((U.shape[1], 1))

    # Calculate parameters using train set and then calculate output and accuracy using test set
    W = np.dot(np.dot(np.linalg.inv(np.dot(U, U.T)), U), G)

    acc = np.count_nonzero(np.dot(U.T, W) > 0) / U.shape[1]

    print('Accuracy = {}%'.format(acc*100))

    plot_data(X1, X2, W)


if __name__ == '__main__':

    X1 = np.array(pd.read_csv('./data/t3.1_class1.csv', names=['feature1', 'feature2']))
    X2 = np.array(pd.read_csv('./data/t3.1_class2.csv', names=['feature1', 'feature2']))

    visualize_data_2(X1, X2)
    plt.title('Data')
    plt.show()

    # Calculate parameters and accuracy on the train and test sets
    desired_output(X1, X2)
