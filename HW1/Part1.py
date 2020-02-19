# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:19:40 2020

@author: Owner
"""
import HW1code as hw
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(17)
A = np.loadtxt("S1train.csv", delimiter = ",")
B = np.loadtxt("S2train.csv", delimiter = ",")
C = np.loadtxt("S1test.csv", delimiter = ",")
D = np.loadtxt("S2test.csv", delimiter = ",")

graph_k = np.linspace(1, 19, 10)
graph_val_mean = []
graph_train_mean = []
graph_gen_mean = []
for i in range(10):    
    a, b, c = hw.jfoldvalidation(hw.knn, A, A[:, 0], hw.misclassified, True, k = 2*i +1, j = 10)
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
plt.title("Error rate for S2train data")
plt.legend()
plt.show()

print(hw.everything(hw.knn, C, A, C[:, 0], hw.misclassified, k = 3))
print(hw.everything(hw.knn, D, B, D[:, 0], hw.misclassified, k = 3))

