# -*- coding: utf-8 -*-
import numpy as np
import statistics as stats
def distance(v1, v2): #measures distance between two vectors
    return np.sqrt(np.sum((v2 - v1) ** 2))
def distancetoall(v1, data): 
    #inputs: vector and a dataset
    #output: euclidean distance from vector to every point in dataset
    distances = []
    for row in data:        
        distances.append(distance(v1, row[1:]))
    return distances
def knn(k, goal, data):
    #inputs: integer k, vector, and a dataset
    #outputs: indices k-nearest neighbors of vector in the dataset
    knn = []
    distances = distancetoall(goal, data)
    for i in range(k):
        index = distances.index(min(distances))
        knn.append(index)
        distances.pop(index)
    return knn
def majorityclass(k, goal, data):
    #inputs: integer k, vector, dataset
    #output: predicted class of vector based on knn
    knearest = knn(k, goal, data)
    classes = []
    for entry in knearest:
        classes.append(data[entry][0])
    return stats.mode(classes)
def defineall(k, data1, data2):
    #inputs: k, dataset, second dataset
    #outputs: predicted classes of vectors in first dataset based on knn of second dataset
    classes = []
    for entry in data1:
        classes.append(majorityclass(k, entry[1:], data2))
    classes2 = np.array(classes)
    classes2 = np.vstack(classes2)
    return classes2
def misclassified(data, labels):
    # input: vector of predictions, vector of truths
    # output: misclassification rate
    misclassified = 0
    i =0
    for row in data:
        if row[0] != labels[i]:
            misclassified += 1 
        i += 1
    return misclassified/np.size(data, 0)
def everything(classifier, data1, data2, labels, metric, k = 1):
    #inputs: classifier (knn), two datasets, ground truths, error metric, k
    #outputs: misclassification rate of the data1 classifier trained on data2
    if classifier == knn:
        predictions = defineall(k, data1, data2)
    if metric == misclassified:
        return misclassified(predictions, labels)
def jfoldvalidation(classifier, data, labels, metric, trainingerror, k = 1, j=3):
    np.random.seed(18)
    np.random.shuffle(data)
    validation_error = []
    training_error = []
    gen_error = []
    for i in range(j):
        test = data[i::j, :].copy()
        training = data.copy()
        training = np.delete(training, slice(i, None, j), 0)
        validation_error.append(everything(classifier, test, training, test[:, 0], metric, k))
        if trainingerror:
            training_error.append(everything(classifier, training, training, training[:, 0], metric, k))
            #training_error.append(everything(classifier, training, training, ))
    if trainingerror:
        for i in range(j):
            gen_error.append(validation_error[i] - training_error[i])
        return validation_error, training_error, gen_error
    else:
        return validation_error
def meanperformance(val_error , train_error = [0], gen_error = [0]):
    val_mean = stats.mean(val_error)
    val_25 = np.percentile(val_error, 25)
    val_75 = np.percentile(val_error, 75)
    train_mean = stats.mean(train_error)
    gen_mean = stats.mean(gen_error)
    return val_mean, train_mean, gen_mean