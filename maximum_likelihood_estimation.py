'''

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from visualization import visualize_data_2

def read_data(class1_path, class2_path):
    cl1 = pd.read_csv('./data/t3.2_class1.csv', names=['feature1', 'feature2'])
    cl2 = pd.read_csv('./data/t3.2_class2.csv', names=['feature1', 'feature2'])

    X1 = np.array(cl1)
    X2 = np.array(cl2)
    visualize_data_2(X1, X2)
    plt.title('Data')
    plt.show()

    X_ = np.concatenate((X1, X2), axis=0)
    m = X1.shape[0]
    n = X1.shape[1]

    classes_ = np.zeros(2 * m)
    classes = np.zeros(2 * m)
    classes_[:m] = np.ones((m))
    X = np.zeros((2 * m, n))
    ind = np.random.choice(2 * m, m, replace=False)
    X[:m] = X_[ind, :]
    X[m:] = np.delete(X_, ind, axis=0)
    classes[:m] = classes_[ind]
    classes[m:] = np.delete(classes_, ind)

    return X, classes


def mle(X, L, epochs):
    m = X.shape[0]
    n = X.shape[1]

    P = np.zeros((L, 1))
    M = np.zeros((L, n))
    S = np.zeros((L, n, n))

    # Initialization
    q = np.zeros((m, L))
    f = np.zeros((m, L))
    # Probabilities for class1
    q[:m // 2, 0] = np.random.rand() * 0.5 + 0.5
    q[:m // 2, 1] = 1 - q[:m // 2, 0]

    # Probabilities for class2
    q[m // 2:, 1] = np.random.rand() * 0.5 + 0.5
    q[m // 2:, 0] = 1 - q[m // 2:, 1]

    for e in range(epochs):
        for i in range(L):
            P[i] = q[:, i].sum() / m
            M[i, :] = np.dot(q[:, i], X) / m
            for j in range(m):
                x = np.reshape(X[j, :] - M[i], (1, 2))
                S[i, :, :] += (q[j, i]) * np.dot(x.T, x)
            S[i, :] = S[i, :] / m
            for j in range(m):
                f[j, i] = 1 / (2 * np.pi * np.sqrt(np.linalg.det(S[i]))) * np.exp(
                    1 / 2 * np.dot(np.dot((X[j, :] - M[i, :]), S[i, :]), (X[j, :] - M[i, :]).T))
        for i in range(L):
            q[:, i] = (P[i] * f[:, i]) / np.dot(P.T, f.T)

    return q


if __name__ == '__main__':

    class1_path = './data/t3.2_class1.csv'
    class2_path = './data/t3.2_class1.csv'
    print('Processing...')
    X, classes = read_data(class1_path, class2_path)

    q = mle(X, 2, 250)
    print('done')
    m = classes.shape[0]
    TP = 0
    for i in range(m):
        if q[i, 0] > q[i, 1]:
            cl = 0.
        else:
            cl = 1.
        if cl == classes[i]:
            TP += 1
    accuracy = TP / m
    print('Accuracy = {}%'.format(accuracy*100))
