'''
Classification of data with known probability density function using Bayes theorem
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from visualization import visualize_data_2


def calc_normal_dist(X, M, S):
    '''
    :param X: numpy array, data
    :param M: numpy array, mean value for all featires
    :param S: numpy array, covariance matrix
    :return: f: list, probability density function for input X
    '''

    m = X.shape[0]
    n = X.shape[1]
    M = np.reshape(M, newshape=(n,1))

    f = []
    for i in range(m):
        x = np.reshape(X[i,:].T, newshape=(n,1))
        f.append(np.squeeze(1/2 * (np.dot(np.dot((x - M).T, np.linalg.inv(S)), (x - M)) + np.log(np.linalg.det(S)))))

    return np.array(f)

def generate_data(m, P11, P12, P21, P22, feat_stat):
    '''
    :param m: int, number of samples
    :param P11, P12, P21, P22: float, probabilities for all distributions
    :param feat_stat: dict with mean values and std for data
    :return:
    '''

    X1 = P11 * np.random.multivariate_normal(feat_stat['M11'], feat_stat['S11'], size=m) + P12 * np.random.multivariate_normal(feat_stat['M12'], feat_stat['S12'], size=m)
    X2 = P21 * np.random.multivariate_normal(feat_stat['M21'], feat_stat['S21'], size=m) + P22 * np.random.multivariate_normal(feat_stat['M22'], feat_stat['S22'], size=m)

    return X1, X2

def wald(X, a, b, feat_stat):
    '''
    :param X: numpy array, data
    :param a, b: float
    :param feat_stat:
    :return: dict with mean values and std for data
    '''

    m = X.shape[0]
    n = X.shape[1]
    sm1 = []
    sm2 = []
    for k in range(m):
        sm = []
        h = 0
        for l in range(k, m):
            x = np.reshape(X[l,:], newshape=(1,n))
            h += np.squeeze(calc_normal_dist(x, feat_stat['M11'], feat_stat['S11']) + calc_normal_dist(x, feat_stat['M12'], feat_stat['S12'])
                            - calc_normal_dist(x, feat_stat['M21'], feat_stat['S21']) - calc_normal_dist(x, feat_stat['M22'], feat_stat['S22']))
            sm.append(h)
            if h < a:
                sm1.append(np.array(sm))
                break
            elif h > b:
                sm2.append(np.array(sm))
                break
    return np.array(sm1), np.array(sm2)

def plot_decision_boundary(X1, X2, feat_stat, thresh):
    '''
    :param X1: numpy array, class1 data
    :param X2: numpy array, class2 data
    :param feat_stat: dict with mean values and std for data
    :param thresh: float
    '''

    X_f = np.concatenate((X1, X2))
    visualize_data_2(X1, X2)
    x_min, x_max = X_f[:, 0].min() - 1, X_f[:, 0].max() + 1
    y_min, y_max = X_f[:, 1].min() - 1, X_f[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x1 = xx.ravel()
    x2 = yy.ravel()
    X = np.concatenate((np.reshape(x1, (x1.shape[0], 1)), np.reshape(x2, (x2.shape[0], 1))), axis=1)

    h = calc_normal_dist(X, feat_stat['M11'], feat_stat['S11']) + calc_normal_dist(X, feat_stat['M12'], feat_stat['S12']) \
      - calc_normal_dist(X, feat_stat['M21'], feat_stat['S21']) - calc_normal_dist(X, feat_stat['M22'], feat_stat['S22'])

    Y = np.array((h > thresh).astype(int))
    Y = np.reshape(Y, (xx.shape[0], xx.shape[1]))
    plt.contourf(xx, yy, Y, cmap=plt.cm.Spectral)


if __name__ == '__main__':

    # Define parameters for data
    P11 = 0.6
    P12 = 0.4
    P21 = 0.55
    P22 = 0.45

    M11 = np.array([1, 1])
    M12 = np.array([6, 4])
    S11 = np.array([[4, 1.1], [1.1, 2]])
    S12 = np.array([[3, -0.8], [-0.8, 1.5]])

    M21 = np.array([7, -4])
    M22 = np.array([6, 0])
    S21 = np.array([[2, 1.1], [1.1, 4]])
    S22 = np.array([[3, -0.8], [-0.8, 0.5]])

    # Generate data with defined parameters
    m = 500
    feat_stat = {'M11': M11, 'S11': S11, 'M12': M12, 'S12': S12, 'M21': M21, 'S21': S21, 'M22': M22, 'S22': S22}
    X1, X2 = generate_data(m, P11, P12, P21, P22, feat_stat)
    visualize_data_2(X1, X2)
    plt.title('Data')
    plt.show()

    # Bayesian classifier
    thresh = np.log((P11*P12)/(P21*P22))
    h1 = calc_normal_dist(X1, M11, S11) + calc_normal_dist(X1, M12, S12) - calc_normal_dist(X1, M21, S21) - calc_normal_dist(X1, M22, S22)
    h2 = calc_normal_dist(X2, M11, S11) + calc_normal_dist(X2, M12, S12) - calc_normal_dist(X2, M21, S21) - calc_normal_dist(X2, M22, S22)
    accuracy = (np.count_nonzero(h1 < thresh) + np.count_nonzero(h2 > thresh)) / (2*m)
    error1 = np.count_nonzero(h1 > thresh)
    error2 = np.count_nonzero(h2 < thresh)
    print('Number of errors for class 1 = ', error1)
    print('Number of errors for class 2 = ', error2)
    print('Accuracy = {}%\n'.format(accuracy * 100))

    plot_decision_boundary(X1, X2, feat_stat, thresh)
    plt.title('Decision boundary')
    plt.show()

    # Changing the threshold value with values c11, c12, c21, c22 number of the errors for class 1 could be decreased
    c11 = 0
    c22 = 0
    c12 = 12
    c21 = 2
    thresh = np.log((c12-c22)*(P11*P12)/((P21*P22)*(c21-c11)))
    h1 = calc_normal_dist(X1, M11, S11) + calc_normal_dist(X1, M12, S12) - calc_normal_dist(X1, M21, S21) - calc_normal_dist(X1, M22, S22)
    h2 = calc_normal_dist(X2, M11, S11) + calc_normal_dist(X2, M12, S12) - calc_normal_dist(X2, M21, S21) - calc_normal_dist(X2, M22, S22)
    accuracy = (np.count_nonzero(h1 < thresh) + np.count_nonzero(h2 > thresh)) / (2*m)
    error1 = np.count_nonzero(h1 > thresh)
    error2 = np.count_nonzero(h2 < thresh)
    print('Number of errors for class 1 = ', error1)
    print('Number of errors for class 2 = ', error2)
    print('Accuracy = {}%'.format(accuracy * 100))

    plot_decision_boundary(X1, X2, feat_stat, thresh)
    plt.title('Decision boundary')
    plt.show()

    # Implement Wald test
    e1 = 1e-4
    e2 = 1e-5
    A = (1 - e1) / e2
    B = e1 / (1 - e2)
    a = - np.log(A)
    b = - np.log(B)

    sm1, sm2 = wald(X1, a, b, feat_stat)
    for i in range(sm1.shape[0]):
        #x_plot = [i for i in range(sm1[i].shape[0])]
        x_plot = np.arange(0, sm1[i].shape[0]) + 1
        if i == 0:
            plt.plot(x_plot, sm1[i], '-b', label='class1')
        plt.plot(x_plot, sm1[i], '-b')
    for i in range(sm2.shape[0]):
        x_plot = np.arange(0, sm2[i].shape[0]) + 1
        if i == 0:
            plt.plot(x_plot, sm2[i,:], '-r', label='class2')
        plt.plot(x_plot, sm2[i,:], '-r')
    sm1, sm2 = wald(X2, a, b, feat_stat)
    for i in range(sm1.shape[0]):
        x_plot = np.arange(0, sm1[i].shape[0]) + 1
        if i == 0:
            plt.plot(x_plot, sm1[i], '-b', label='class1')
        plt.plot(x_plot, sm1[i], '-b')
    for i in range(sm2.shape[0]):
        x_plot = np.arange(0, sm2[i].shape[0]) + 1
        if i == 0:
            plt.plot(x_plot, sm2[i], '-r', label='class2')
        plt.plot(x_plot, sm2[i], '-r')

    plt.plot([1, 4], [a, a], '-y', label='a')
    plt.plot([1, 4], [b, b], '-g', label='b')
    plt.legend()
    plt.xlabel('m')
    plt.ylabel('sm')
    plt.title('Wald test')
    plt.show()
