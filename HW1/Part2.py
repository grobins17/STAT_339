# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:43:57 2020

@author: Owner
"""
import HW1code as hw
import numpy as np
import matplotlib.pyplot as plt
Train5 = np.loadtxt("Train5.csv", delimiter = ",", skiprows = 1)
Train9 = np.loadtxt("Train9.csv", delimiter = ",", skiprows = 1)
A = np.concatenate((Train5, Train9), axis = 0)
a = np.concatenate((np.ones((600, 1)),-1 * np.ones((600,1))), axis = 0)
A = np.insert(A, [0], a, axis = 1)
Test5 = np.loadtxt("Test5.csv", delimiter = ",", skiprows = 1)
Test9 = np.loadtxt("Test9.csv", delimiter = ",", skiprows = 1)
B = np.concatenate((Test5, Test9), axis = 0)
b =  np.concatenate((np.ones((292, 1)),-1 * np.ones((292, 1))), axis = 0)
B = np.insert(B, [0], b, axis = 1)
graph_k = np.linspace(1, 19, 10)
graph_val_mean = []
graph_train_mean = []
graph_gen_mean = []
for i in range(10):    
    a, b, c = hw.jfoldvalidation(hw.knn, B, B[:, 0], hw.misclassified, True, k = 2*i +1, j = 10)
    e, f, g = hw.meanperformance(a,b,c)
    graph_val_mean.append(e)
    graph_train_mean.append(f)
    graph_gen_mean.append(g)
plt.plot(graph_k, graph_val_mean, label = 'Validation Error Rate')
plt.plot(graph_k, graph_train_mean, label = 'Training Error Rate')
plt.plot(graph_k, graph_gen_mean, label = 'Generalization Error Rate')
plt.xlabel('K')
plt.ylabel('Misclassification rate')
plt.title("Error rate for S1train data")
plt.legend()
plt.show()
plt.cla()
 k = 1
print(hw.everything(hw.knn, B, A, B[:, 0], hw.misclassified, k = 1))
