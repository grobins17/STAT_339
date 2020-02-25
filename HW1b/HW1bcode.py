# -*- coding: utf-8 -*-
"""
@author: Garrett Robins

"""
import numpy as np 
import matplotlib.pyplot as plt 
import math
import statistics as stats
import scipy.stats as sci
def simpleregression(matrix, lambd = 0): #helper function for the polyregression function.
    #inputs: matrix in the form x (N x d), t (N x 1), lambda for ridge regression
    #outputs: regression coefficients (d x 1)
    datacol = np.vstack(matrix[:, :matrix.shape[1]-1])
    ones = np.ones((matrix.shape[0],1))
    X = np.append(ones, datacol, axis = 1)
    hat = np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(X), X) , lambd * np.identity(X.shape[1]))), np.transpose(X))
    w = np.matmul(hat, matrix[:, matrix.shape[1]-1 ])
    return w
def y(x, w):
    y=0
    for i in range(w.shape[0]):
        y += w[i]*(x**i)
    return y
def powercolumns(x, D): 
    if D < 1:
        return
    M = np.zeros((x.shape[0], D))
    for i in range(D):
        M[:, i] = np.hstack(np.power(x, i+1))
    return M
def polyregression(t, pred, D, lambd = 0):
    X = powercolumns(pred, D)
    hat = np.append(X, t, axis = 1)
    return simpleregression(hat, lambd)
def mse(t, yhat):
    mse = 0
    for i in range(len(t)):
        mse += math.pow(t[i] - yhat[i], 2)
    return mse/len(t)
def kfoldvalidation(data, K = 10, seed = 1, D= 1, lambd = 0, trainingerror = False):
    np.random.seed(seed) #sets the seed
    np.random.shuffle(data) #shuffles the data
    validation_error = []
    training_error = []
    for i in range(K): #for K folds
        test = data[i::K, :].copy() #take every Kth element and put it in the test set
        training = data.copy() 
        training = np.delete(training, slice(i, None, K), 0) #take the rest of the data for the training set
        w = polyregression(np.vstack(training[:, training.shape[1]-1]), np.vstack(training[:, :training.shape[1]-1]), D, lambd) #get the OLS coefficients for model trained on training set
        validation_error.append(mse(test[:, test.shape[1]-1], y(test[:, :test.shape[1]-1], w))) #get the MSE of the model on the test set 
        if trainingerror:
            training_error.append(mse(training[:, training.shape[1]-1], y(training[:, :training.shape[1]-1], w)))  
    if trainingerror: 
        return stats.mean(validation_error), stats.stdev(validation_error), stats.mean(training_error), stats.stdev(training_error)
    else:
        return stats.mean(validation_error), stats.stdev(validation_error)
def polychoose(t, x, D, K, seed =1, training_error = False): #TODO: training error 
    t = np.vstack(t)
    x = np.vstack(x)
    data = np.append(x, t, axis =1)
    fold_means = []
    fold_sds = []
    training_means = []
    training_sds = []
    for i in range(D):
        k = kfoldvalidation(data, K, seed, i+1, 0, training_error)
        fold_means.append(k[0])
        fold_sds.append(k[1])
        if training_error:
            training_means.append(k[2])
            training_sds.append(k[3])
    if training_error:
        return fold_means, fold_sds, training_means, training_sds, fold_means.index(min(fold_means)) +1 
    else:
         return fold_means, fold_sds, fold_means.index(min(fold_means)) +1 
A = np.loadtxt("womens100.csv", delimiter = ",")
Year_mean = stats.mean(A[:,0])
Year_sd = stats.stdev(A[:, 0])
time_sd = stats.stdev(A[:, 1])
time_mean = stats.mean(A[:, 1])
A = sci.zscore(A, axis = 0)

w = polyregression(np.vstack(A[:, 0]), np.vstack(A[:,1]), 1)
#same as the one in the textbook
x = np.linspace(-2, 2, 10000)
plt.scatter(A[:,0], A[:, 1])
plt.plot(x, y(x,w), color = "red")
plt.show()
x = y((2012 - Year_mean)/Year_sd, w)
#print((x * time_sd)+ time_mean)
##10.5996
##real was 10.75
x = y((2016 - Year_mean)/Year_sd, w)
#print((x * time_sd)+ time_mean)
##10.5393
##real was 10.71
#TODO: calculate the squared error
B = np.loadtxt("synthdata2016.csv", delimiter = ",")
B = sci.zscore(B, axis = 0)
w = polyregression(np.vstack(B[:,1]), np.vstack(B[:,0]), 4)
plt.cla()
plt.scatter(B[:, 0], B[:, 1])
x = np.linspace(-1.75, 1.75, 1000)
plt.plot(x, y(x, w))
plt.show()

print(kfoldvalidation(B, 10, 1,3, 0, True))



f = polychoose(B[:,1], B[:,0], 10, 50, 1, True)
plt.cla()
x = np.linspace(1, len(f[0]), len(f[0]))
plt.title("Optimal Polynomial for Synthetic Data, K = 50")
plt.plot(x, f[0], color = "blue", label = "Validation Error")
plt.plot(x, f[2], color = "red", label = "Training Error")
plt.errorbar(x, f[0], f[1], color = "blue")
plt.errorbar(x, f[2], f[3], color = "red")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Polynomial Order")
plt.show()

f = polychoose(B[:,1], B[:,0], 10, 10, 1, True)
plt.cla()
x = np.linspace(1, len(f[0]), len(f[0]))
plt.title("Optimal Polynomial for Synthetic Data, K = 10")
plt.plot(x, f[0], color = "blue", label = "Validation Error")
plt.plot(x, f[2], color = "red", label = "Training Error")
plt.errorbar(x, f[0], f[1], color = "blue")
plt.errorbar(x, f[2], f[3], color = "red")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Polynomial Order")
plt.show()


f = polychoose(A[:,1], A[:,0], 6, 10, 1, True)
plt.cla()
x = np.linspace(1, len(f[0]), len(f[0]))
plt.title("Optimal Polynomial for Olympic Data, K = 10")
plt.plot(x, f[0], color = "c", label = "Validation Error")
plt.plot(x, f[2], color = "r", label = "Training Error")
plt.errorbar(x, f[0], f[1], ecolor = "blue")
plt.errorbar(x, f[2], f[3], ecolor = "blue")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Polynomial Order")
plt.show()

f = polychoose(A[:,1], A[:,0], 6, 19, 1, True)
plt.cla()
x = np.linspace(1, len(f[0]), len(f[0]))
plt.title("Optimal Polynomial for Olympic Data, K = 19")
plt.plot(x, f[0], color = "c", label = "Validation Error")
plt.plot(x, f[2], color = "red", label = "Training Error")
plt.errorbar(x, f[0], f[1], ecolor = "blue")
plt.errorbar(x, f[2], f[3], ecolor = "orange")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Polynomial Order")
plt.show()