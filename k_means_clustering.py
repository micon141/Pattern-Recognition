''''
k means clustering algorithm implemented in numpy
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualization import visualize_data_4


def real_mean_values(X1, X2, X3, X4):
    '''
    :param X1, X2, X3, X4: numpy array, data divedide to 4 classes
    :return: mean values for all calsses
    '''
    return (X1.mean(axis=0), X2.mean(axis=0), X3.mean(axis=0), X4.mean(axis=0))

def init_means(X, k):
    '''
    :param X: numpy array, data
    :param k: number of clusters
    :return: mean value
    '''

    np.random.seed(5)
    f = X.shape[1]
    means = np.zeros(shape=(k,f))

    min_features = X.min(axis=0)
    max_features = X.max(axis=0)

    # Initialize mean values
    for i in range(f):
        for n in range(k):
            means[n,i] = np.random.uniform(min_features[i], max_features[i])
    return means

def clust(feature, means):
    '''
    :param feature: numpy array, current sample
    :param means: numpy array, mean values for all calsses
    :return: flaot, class for current sample
    '''
    min_val = 10000
    k = means.shape[0]
    f = means.shape[1]
    for i in range(k):
        dist = np.sum(np.square(feature - means[i,:]))
        if dist < min_val:
            min_val = dist
            ind = i
    return ind

def update_mean(feature, mean, num):
    '''
    :param feature: numpy array, new feature for appropriate class
    :param mean: float, current mean value
    :param num: float, number of samles
    :return: update mean value
    '''
    return (mean * (num-1) + feature) / (num)

def calc_means(X, k, epochs):
    '''
    :param X: numpy array, all data
    :param k: float, number of clusters
    :param epochs: float, number of iteration of the algorithm
    :return:
        mean: numpy array, estimated mean values
        clusters: numpy array, class id for all samples
    '''
    means = init_means(X, k)
    m = X.shape[0]
    clusters = np.zeros((m)) - 1
    for e in range(epochs):
        for i in range(m):
            ind = clust(X[i,:], means)
            clusters[i] = ind
            num = np.count_nonzero(clusters == ind)
            means[ind,:] = update_mean(X[i,:], means[ind,:], num)

    return means, clusters

def sort_means(means):
    '''
    :param means: numpy array, mean values
    :return: numpy array, sorted mean values
    '''

    n = means.shape[0]
    for i in range(n-1):
        for j in range(i+1, n):
            if (means[i][0] < means[j][0]):
                t0 = means[i][0]
                t1 = means[i][1]
                means[i][0] = means[j][0]
                means[i][1] = means[j][1]
                means[j][0] = t0
                means[j][1] = t1
    return means


if __name__ == '__main__':

    # Read data for all four classes
    X1 = np.array(pd.read_csv('./data/t4.1_class1.csv', names=['feature1', 'feature2']))
    X2 = np.array(pd.read_csv('./data/t4.1_class2.csv', names=['feature1', 'feature2']))
    X3 = np.array(pd.read_csv('./data/t4.1_class3.csv', names=['feature1', 'feature2']))
    X4 = np.array(pd.read_csv('./data/t4.1_class4.csv', names=['feature1', 'feature2']))

    # Display data
    visualize_data_4(X1, X2, X3, X4)
    plt.title('Data')
    plt.show()

    # Define matrix with all data
    X = np.array([])
    X = np.append(X1, X2, axis=0)
    X = np.append(X, X3, axis=0)
    X = np.append(X, X4, axis=0)

    # Calculate real mean values for all classes
    m1, m2, m3, m4 = real_mean_values(X1, X2, X3, X4)
    print('Real mean values:\n{}\n{}\n{}\n{}\n'.format(m1, m2, m3, m4))

    k = 4 # number of clusters
    means, clusters = calc_means(X, k, 10)
    means = sort_means(means)
    print('Estimated mean values:')
    for i in range(k):
        print(means[i])

    # Calculate accuracy
    m = X.shape[0]
    accuracy = 0
    acc = []
    for i in range(k):
        for j in range(k):
            acc.append(np.count_nonzero(clusters[i*m//k:(i+1)*m//k] == j))
        accuracy += max(acc)
    accuracy = accuracy / m
    print('Accuracy = {}%'.format(accuracy*100))

    # Display data and dots for mean values
    visualize_data_4(X1, X2, X3, X4)
    for i in range(means.shape[0]):
        plt.plot(means[i,0], means[i,1], 'ok')
    plt.title('Data with estimated mean values')
    plt.show()
