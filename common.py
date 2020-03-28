import numpy as np
import matplotlib.pyplot as plt

def split_data(X, thr=0.85):
    '''
    :param X: numpy array
    :param thr: float, value for splitting dataset
    :return:
    '''
    m = X[0].shape[0]
    # Number of training examples
    m_train = round(thr * m)
    # Define arrays for test and train datasets
    X_train = []
    X_test = []
    # Define training and test sets for each feature
    for i in range(len(X)):
        X_train.append(X[i][0:m_train,:])
        X_test.append(X[i][m_train:,:])

    return X_train, X_test

def linear_classifier(X1, X2):
    '''
    :param X1: numpy array, class1 samples
    :param X2: numpy array, class2 samples
    :return:
        V, vo: parameters of the calssifier
        accuracy: float, accuracy of the classifier
    '''

    #Split data for each class
    X_train, X_test = split_data([X1, X2])
    X1_train = X_train[0]
    X2_train = X_train[1]
    X1_test = X_test[0]
    X2_test = X_test[1]

    n = X1.shape[1] # number of features
    m = X1.shape[0] # number of samles
    # Calculate mean values and std
    M1 = X1.mean(axis=0)
    M2 = X2.mean(axis=0)
    X1_feat = []
    X2_feat = []
    for i in range(n):
        X1_feat.append(X1[:,i])
        X2_feat.append(X2[:,i])
    S1 = np.cov(X1_feat)
    S2 = np.cov(X2_feat)

    errors = 2*m
    all_errors = []
    s_values = np.linspace(0, 1, num=20)
    # Linear classifier
    for s in s_values:
        # Calculate V = (sS1 + (1-s)S2)^-1 (M2 - M1)
        V = np.dot(np.linalg.inv(s*S1 + (1-s)*S2), (M2 - M1))
        V = np.reshape(V, (n,1))
        Y1 = np.dot(V.T, X1_train.T)
        Y2 = np.dot(V.T, X2_train.T)

        for v0 in np.linspace(-max(Y1.max(), Y2.max()), -min(Y1.min(), Y2.min()), num = 50):
            ers = (Y1 > - v0).sum() + (Y2 < - v0).sum()
            if ers < errors:
                errors = ers
                vopt = v0

        Y1_test = np.dot(V.T, X1_test.T)
        Y2_test = np.dot(V.T, X2_test.T)
        errors = (Y1 > - vopt).sum() + (Y2 < - vopt).sum()
        all_errors.append(errors)

    sopt = s_values[all_errors.index(min(all_errors))] # Find index of the minimum value in list of all errors

    plt.plot(s_values, all_errors)
    plt.title('errors(s)')
    plt.xlabel('s')
    plt.show()

    print('Optimal value for s = ', sopt)

    # Calculate parameters for optimal s
    V = np.dot(np.linalg.inv(sopt*S1 + (1-sopt)*S2), (M2 - M1))
    V = np.reshape(V, (n,1))
    Y1 = np.dot(V.T, X1.T)
    Y2 = np.dot(V.T, X2.T)

    for v0 in np.linspace(-max(Y1.max(), Y2.max()), -min(Y1.min(), Y2.min()), num = 50):
        ers = (Y1 > - v0).sum() + (Y2 < - v0).sum()
        if ers < errors:
            errors = ers
            vopt = v0
    v0 = vopt

    accuracy = (2*m - errors) / (2*m)

    return V, v0, accuracy
