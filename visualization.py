'''
Data visualization using matplotlib
'''

import matplotlib.pyplot as plt

def visualize_data_2(X1, X2):
    plt.plot(X1[:,0], X1[:,1], '.b', label='class1')
    plt.plot(X2[:,0], X2[:,1], '*r', label='class2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

def visualize_data_4(X1, X2, X3, X4):
    plt.plot(X1[:,0], X1[:,1], 'ob', label='class1')
    plt.plot(X2[:,0], X2[:,1], '*r', label='class2')
    plt.plot(X3[:,0], X3[:,1], '.y', label='class3')
    plt.plot(X4[:,0], X4[:,1], '^g', label='class4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
