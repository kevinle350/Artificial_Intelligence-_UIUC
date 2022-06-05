# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np
import math
from collections import Counter

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    #print(train_set)
    '''
        initialize weights arr with 0
        bias parameter is just w0x0
        max_iter is the number of epochs 
    '''
    
    weights = np.zeros(train_set.shape[1])
    b = 0

    for i in range(max_iter):
        for features, label in zip(train_set, train_labels):
            func = np.dot(features, weights)+b
            if func > 0:
                y = 1
            else:
                y = 0
            weights = weights + learning_rate*(label-y)*features
            b = b + learning_rate*(label-y)
    
    return weights, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    trained_weights, trained_bias = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    dev_labels = []
    for image in dev_set:
        func = np.dot(image, trained_weights)+trained_bias
        if func > 0:
            y = 1
        else:
            y = 0    
        dev_labels.append(y)
    return dev_labels

def mode(lyst):
    cnt = Counter(lyst)
    return cnt[True] > cnt[False]


def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    '''
        find most similar training example and copy label
        compute the distance from X(input img) to every stored image
    '''
    preds = np.zeros(len(dev_set))

    for dIndex, dImage in enumerate(dev_set):
        distLabel = []
        for tIndex, tImage in enumerate(train_set):
            dist = np.linalg.norm(tImage - dImage)
            distLabel.append((train_labels[tIndex], dist))
        distLabel.sort(key= lambda tup: tup[1])

        kLabels = [i[0] for i in distLabel[:k]]
        preds[dIndex] = mode(kLabels)

    return list(preds)