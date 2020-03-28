'''
Linear classification algorithm
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from visualization import *
from common import linear_classifier


if __name__ == '__main__':

    cl1 = pd.read_csv('./data/t3.1_class1.csv', names=['feature1', 'feature2'])
    cl2 = pd.read_csv('./data/t3.1_class2.csv', names=['feature1', 'feature2'])
    X1 = np.array(cl1)
    X2 = np.array(cl2)

    visualize_data_2(X1, X2)
    plt.title('Data')
    plt.show()

    # Implement linear classifier for two classes
    V, v0, acc = linear_classifier(X1, X2)

    visualize_data_2(X1, X2)
    # Plot decision boundary
    x_pl = [X1[:,0].min()-1, X1[:,0].max()+1]
    y_pl = -1/V[1] * (v0 + V[0]*x_pl)
    plt.plot(x_pl, y_pl, '-g', label='Decision boundary')
    plt.legend()
    plt.show()

    print('Accuracy = {}%'.format(acc*100))
