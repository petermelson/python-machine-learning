# ======================================================================================
#
# Project:          Python Machine Learning - Perceptron
# Description:      Implementation of a simple perceptron in Python
# Author:           Peter Melson
# Version:          0.1
# Date:             December 2023
#
# ======================================================================================

import logging
import numpy as np


# --------------------------------------------------------------------------------------
# Perceptron Class
# --------------------------------------------------------------------------------------
class Perceptron(object):
    """
    Perceptron class

    Parameters:
    -----------
    :param eta: (float) learning rate
    :param n_iter: (int) number of iterations
    :param random_state: (int) random number generator seed for random weight initialisation

    Attributes:
    -----------
    w_: (array) Weights after fitting
    errors_: (list) Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        Constructor for perceptron class.

        :param eta:
        :param n_iter:
        :param random_state:
        """

        # Initialisation.
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the perceptron to training data.

        :param X:
        :param y:
        :return: (object)
        """

        # Initialisation.
        rgen = np.random.RandomState(seed=self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # Perform fitting.
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate the net put i.e. input after application of weightings.

        :param X:
        :return:
        """

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Return class label after unit step.
        :param X:
        :return:
        """

        return np.where(self.net_input(X) >= 0.0, 1, -1)
