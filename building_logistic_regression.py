# -*- coding: utf-8 -*-
"""Building Logistic Regression.ipynb

Original file is located at
    https://colab.research.google.com/drive/1RjvzveAQGTPdADF_m1CaTwFlYbMRc5OK

JAI MAA SARASWATI
"""

import numpy as np

class Logistic_Regression():
  # declaring learning rate and number of iterations(hyper parameters)
  def __init__(self,):
    self.learning_rate=learning_rate
    self.no_of_iterations=no_of_iterations

  # fit function to train the model withdataset
  def fit(self ,X,Y):
    self.m , self.n=X.shape
    self.w=np.zeros(self.n)
    self.b=0
    self.X=X
    self.Y=Y
    # implementing gradient descent for optimization
    for i in range(no_of_iterations):
      self.update_weights()

  def update_weights(self):
    # Y_hat formula
    Y_hat=1/(1+np.exp(-(self.X.dot(self.w) + self.b)))
    # dJ/dW
    dw=(1/self.m)*np.dot(self.X.T,(Y_hat - self.Y))   # x =(769 X 8) y=(769 X 1)
      # x(transpose)=(8 X 769) y=(769 X 1)
    # dJ/db
    db=(1/self.m)*np.sum(Y_hat - self.Y)

    # updating the weights & bias using gradient descent
    self.w =self.w - self.learning_rate * dw

    self.b =self.b - self.learning_rate * db

  # sigmoid equation and decision boundary
  def predict(self):
    Y_pred=1/(1+np.exp(-(self.X.dot(self.w) + self.b)))
    Y_pred=np.where(Y_pred > 0.5 , 1 ,0)
    return Y_pred
