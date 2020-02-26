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
def polychoose(t, x, D, K, seed =1, training_error = False):  
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
def gridsearch(data, D, seed = 1):
    mse = np.zeros((D, 20))
    for l in range(20):
        for i in range(D):
            mse[i , l] = kfoldvalidation(data, 10, seed, i+1, l)[0]
    indices = np.where(mse == np.amin(mse))
    print(indices[0], indices[1])
    print(mse[indices[0], indices[1]])
    return indices[0], indices[1]
def truetestset(t, pred, D, lambd = 0):
    se2012 = []
    se2016 = []
    for i in range(D):
        w = polyregression(t, pred, i+1, lambd)
        twelve = y((2012 - Year_mean)/Year_sd, w)
        sixteen = y((2016 - Year_mean)/Year_sd, w)
        se2012.append(math.pow((-1.09) - twelve, 2))
        se2016.append(math.pow((-1.18) - sixteen, 2))
    return se2012, se2016

A = np.loadtxt("womens100.csv", delimiter = ",")
global Year_mean
Year_mean = stats.mean(A[:,0])
global Year_sd
Year_sd = stats.stdev(A[:, 0])
global time_sd 
time_sd = stats.stdev(A[:, 1])
global time_mean 
time_mean = stats.mean(A[:, 1])
A = sci.zscore(A, axis = 0)
w = polyregression(np.vstack(A[:, 0]), np.vstack(A[:,1]), 1)
#same as the one in the textbook
x = np.linspace(-2, 2, 10000)
plt.scatter(A[:,0], A[:, 1])
plt.plot(x, y(x,w), color = "red")
plt.xlabel("Standardized Years")
plt.ylabel("Standardized Times")
plt.title("Olympic Times vs. Years")
plt.show()
x = y((2012 - Year_mean)/Year_sd, w)
#print((x * time_sd)+ time_mean)
##10.5996 
##real was 10.75 -> z score of -1.0901552042195117
x = y((2016 - Year_mean)/Year_sd, w)
#print((x * time_sd)+ time_mean)
##10.5393
##real was 10.71 -> z score of -1.1826237260059862

B = np.loadtxt("synthdata2016.csv", delimiter = ",")
B = sci.zscore(B, axis = 0)
w = polyregression(np.vstack(B[:,1]), np.vstack(B[:,0]), 4)
plt.cla()
plt.scatter(B[:, 0], B[:, 1])
x = np.linspace(-1.75, 1.75, 1000)
plt.xlabel("Standardized X")
plt.ylabel("Standardized Y")
plt.title("Fitted Synthetic Data")
plt.plot(x, y(x, w))
plt.show()

f = polychoose(B[:,1], B[:,0], 10, 50, 1, True)
plt.cla()
x = np.linspace(1, len(f[0]), len(f[0]))
plt.title("Optimal Polynomial for Synthetic Data, K = 50")
plt.errorbar(x, f[0], f[1], ecolor = "black", color = "blue", label = "Validation Error")
plt.errorbar(x, f[2], f[3], ecolor = "magenta", color = "red", label = "Training Error")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Polynomial Order")
plt.show() #optimal polynomial is 4th order

f = polychoose(B[:,1], B[:,0], 10, 10, 1, True)
plt.cla()
x = np.linspace(1, len(f[0]), len(f[0]))
plt.title("Optimal Polynomial for Synthetic Data, K = 10")
plt.errorbar(x, f[0], f[1], ecolor = "black", color = "blue", label = "Validation Error")
plt.errorbar(x, f[2], f[3], ecolor = "magenta", color = "red", label = "Training Error")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Polynomial Order")
plt.show() #optimal is still 4th order


f = polychoose(A[:,1], A[:,0], 6, 10, 1, True)
plt.cla()
x = np.linspace(1, len(f[0]), len(f[0]))
plt.title("Optimal Polynomial for Olympic Data, K = 10")
plt.errorbar(x, f[0], f[1], ecolor = "black", color = "blue", label = "Validation Error")
plt.errorbar(x, f[2], f[3], ecolor = "magenta", color = "red", label = "Training Error")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Polynomial Order")
plt.show() #second order poly

f = polychoose(A[:,1], A[:,0], 6, 19, 1, True)
plt.cla()
x = np.linspace(1, len(f[0]), len(f[0]))
plt.title("Optimal Polynomial for Olympic Data, K = 19")
#plt.plot(x, f[0], color = "c", label = "Validation Error")
#plt.plot(x, f[2], color = "red", label = "Training Error")
plt.errorbar(x, f[0], f[1], ecolor = "black", color = "blue", label = "Validation Error")
plt.errorbar(x, f[2], f[3], ecolor = "magenta", color = "red", label = "Training Error")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Polynomial Order")
plt.show() #second order poly

gridsearch(A, 20) #optimal is reached with 1st order with lambda = 2 
gridsearch(B, 20) #optimal is reached with 4th order polnomial with lambda = 0

b, c = truetestset(np.vstack(A[:, 1]), np.vstack(A[:,0]), D = 10)
print(b.index(min(b))+1, min(b)) #suggests a 6th order poly
print(c.index(min(c))+1, min(c)) #suggests a 2nd order poly

d, e = truetestset(np.vstack(A[:, 1]), np.vstack(A[:,0]), 1, 2)
print(d, e) #our suggested model does better!

