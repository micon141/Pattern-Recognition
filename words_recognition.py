import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from math import ceil, pi
from mpl_toolkits.mplot3d import Axes3D


def split_data(features):
    train_features = {'A': [], 'E': [], 'I': [], 'O': [], 'U': []}
    test_features = {'A': [], 'E': [], 'I': [], 'O': [], 'U': []}
    for key in features.keys():
        train_features[key] = features[key][:100, :]
        test_features[key] = features[key][100:, :]

    return train_features, test_features


def eliminate_white_parts(img):
    """
    Args:
      img: numpy array, Input image
    Returns:
      img: numpy array, Image with only vowel on it
    """

    im_height = img.shape[0]
    im_width = img.shape[1]
    img = img[40:im_height - 40, 40:im_width - 40]
    im_height = img.shape[0]
    im_width = img.shape[1]

    top = 0
    left = 0
    bottom = im_height
    right = im_width

    # Eliminate whiteness
    height_thres = im_height - 3
    width_thres = im_width - 3
    for i in range(0, im_height):
        if sum(img[i, :]) > width_thres:
            top += 1
        else:
            break

    for i in range(0, im_width):
        if sum(img[:, i]) > height_thres:
            left += 1
        else:
            break

    for i in range(im_height - 1, 0, -1):
        if sum(img[i, :]) > width_thres:
            bottom -= 1
        else:
            break

    for i in range(im_width - 1, 0, -1):
        if sum(img[:, i]) > height_thres:
            right -= 1
        else:
            break

    return img[top:bottom, left:right]


def get_features(img_path):
    img = cv.imread(img_path, 0)
    img = img / 255
    img = cv.threshold(img, 0.5, 1, cv.THRESH_BINARY)[1]

    img = eliminate_white_parts(img)
    im_height, im_width = img.shape

    # Get features:
    x1 = img[ceil(0.4 * im_height):ceil(0.6 * im_height), ceil(0.4 * im_width):ceil(0.6 * im_width)].flatten().mean()
    x2 = img[:, :ceil(im_width / 6)].flatten().mean()
    x3 = img[ceil(0.8 * im_height):, :].flatten().mean()
    x4 = img[:ceil(0.3 * im_height), :].flatten().mean() - img[ceil(0.7 * im_height):, :].flatten().mean()
    x5 = img[ceil(0.1 * im_height):ceil(0.4 * im_height), ceil(0.3 * im_width):ceil(0.7 * im_width)].flatten().mean() - \
         img[ceil(0.6 * im_height):ceil(0.9 * im_height), ceil(0.3 * im_width):ceil(0.7 * im_width)].flatten().mean()
    #x5 = img[:, :ceil(0.3 * im_width)].flatten().mean() - img[:, ceil(0.7 * im_width):].flatten().mean()

    return [x1, x2, x3, x4, x5]


def calc_conf_mat(test_features, train_stat):

    conf_mat = np.zeros((5, 5))
    for ind in range(test_features['A'].shape[0]):
        for i_stat, key_stat in enumerate(['A', 'E', 'I', 'O']):
            f_arr = []
            for i_test, key_test in enumerate(['A', 'E', 'I', 'O']):
                f = 1 / (pi ** 1.5 * np.linalg.det(train_stat[key_stat]['cov']) ** 0.5) * np.exp(-0.5 * \
                 np.dot(np.dot(test_features[key_test][ind,:] - train_stat[key_stat]['mean'], \
                 np.linalg.inv(train_stat[key_stat]['cov'])),test_features[key_test][ind, :] - \
                 train_stat[key_stat]['mean']))

                f_arr.append(f)
            key_ind = f_arr.index(max(f_arr))
            conf_mat[i_stat, key_ind] += 1
    return conf_mat


def print_conf_mat(conf_mat):
    k = conf_mat.shape[0]
    for i in range(k):
        for j in range(k):
            print(format(int(conf_mat[i, j])), end='\t')
        print('\n')
    accuracy = np.trace(conf_mat) / np.sum(conf_mat.flatten())
    print('Accuracy: ', accuracy)


if __name__ == "__main__":

    # Get image paths
    images_folder = 'E:\\Datasets\\vowels'
    all_images = os.listdir(images_folder)
    print("Number of images: {}".format(len(all_images)))
    vowels = {'A':[], 'E':[], 'I':[], 'O':[], 'U':[]}
    for vowel in all_images:
        if 'A' in vowel:
            vowels['A'].append(vowel)
        elif 'E' in vowel:
            vowels['E'].append(vowel)
        elif 'I' in vowel:
            vowels['I'].append(vowel)
        elif 'O' in vowel:
            vowels['O'].append(vowel)
        elif 'U' in vowel:
            vowels['U'].append(vowel)

    # Create dictionary for features for all classes
    print("\nExtracting features...")
    features = {'A':[], 'E':[], 'I':[], 'O':[], 'U':[]}
    for key in features.keys():
        for img in vowels[key]:
            img_path = os.path.join(images_folder, img)
            feat = get_features(img_path)
            features[key].append(np.array(feat))
        features[key] = np.array(features[key])

    # Split data into train and test sets
    train_features, test_features = split_data(features)

    train_stat = {'A':{'mean':np.nan, 'cov':np.nan},
                  'E':{'mean':np.nan, 'cov':np.nan},
                  'I':{'mean':np.nan, 'cov':np.nan},
                  'O':{'mean':np.nan, 'cov':np.nan},
                  'U':{'mean':np.nan, 'cov':np.nan}}

    for key in train_stat.keys():
        train_stat[key]['mean'] = np.mean(train_features[key], axis=0)
        train_stat[key]['cov'] = np.cov(train_features[key], rowvar=False)

    conf_mat_train = calc_conf_mat(train_features, train_stat)
    conf_mat_test = calc_conf_mat(test_features, train_stat)

    print('Train Confusion Matrix:\n')
    print_conf_mat(conf_mat_train)
    print('\nTest Confusion Matrix:\n')
    print_conf_mat(conf_mat_test)

