# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:17:04 2020

@author: pozir
"""
import pandas as pd
import csv
import random
import math
import operator

lines = csv.reader(open(r'data/pima-indians-diabetes.csv'))

# Create a list of lists out of a reader object:
    
def data_transformer(reader_object):
    dataset = list(reader_object)
    
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    
    return dataset

df = data_transformer(lines)

# Split data in train and test set:
    
def train_test_split(data, split_ratio):
    
    test_number = round(len(data)*split_ratio)
    indexes = [i for i in range(0, len(data))]
    
    test_ind = random.choices(indexes, k = test_number)
    train_ind = [i for i in indexes if i not in test_ind]
    
    test_set = [data[i] for i in test_ind]
    train_set = [data[i] for i in train_ind]
    
    return train_set, test_set

train, test = train_test_split(df, 0.20)

# Create a dictionary with the keys represented by the classes we are to 
# predict and their corresponding instances as their values:

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
           
            separated[vector[-1]] = []
           
        separated[vector[-1]].append(vector)
    return separated

# The functions that calculates mean and standard deviation of a list of 
# numbers:

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    st_dev = math.sqrt(sum([(i - avg)**2 for i in numbers])/float(len(numbers)-1))
    return st_dev

# Create the summaries of tuples for each attribute of the dataset:
    
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

# Apply the separateByClass and summarize functions created above on the train 
# set to obtain the dictionary of mean and standard deviation for each variable
# by each class value:
    
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

summaries = summarizeByClass(train)

# Calculate the Gaussian probability density function:
    
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# For each instance create the dictionary of probabilites of the classes 1
# and 0 and their values by applying the function calculateProbability to 
# each value of the vector/instance using corresponding mean and standard 
# deviation from the summaries dictionary and ultimately multiplying the 
# individual probability results within each class:

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    return probabilities

# Extract the key with the highest value from the dictionary of the probabili-
# ties created by the calculateClassProbabilities for each instance:

def predict(summaries, inputVector):
    
    probabilities = calculateClassProbabilities(summaries, inputVector) 
    label = max(probabilities.items(), key=operator.itemgetter(1))[0]

    return label
    
# Iterate over test set and get the probabilities of each instance applying
# the predict function created above:
    
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

predictions = getPredictions(summaries, test)

# Get accuracy:

def get_accuracy(preds, test_set):
    
    correct = []
    
    for i in range(len(test_set)):
        if test[i][-1] == preds[i]:
            correct.append(1)
        else:
            correct.append(0)
            
    return sum(correct)/float(len(test_set))*100

get_accuracy(predictions, test)

# Precision per class:

def precision_per_class(preds, test_set, class_):
    
    correct = []
    class_positive = []
    
    for i in range(len(test_set)):
        
        if (test_set[i][-1] == class_) and (preds[i] == class_):
            correct.append(class_)
            class_positive.append(class_)
        
        elif (test_set[i][-1] == class_) and (test_set[i][-1] != preds[i]):
            class_positive.append(class_)
        
    return len(correct)/len(class_positive)

precision_per_class(predictions, test, 0) 
precision_per_class(predictions, test, 1)   
    
def recall_per_class(preds, test_set, class_):
    
    correct = []
    class_predictions = []
    
    for i in range(len(test_set)):
        
        if (preds[i] == class_) and (test_set[i][-1] == class_):
            correct.append(class_)
            class_predictions.append(class_)
        
        elif (preds[i] == class_) and (test_set[i][-1] != class_):
            class_predictions.append(class_)
        
    return len(correct)/len(class_predictions)
          
recall_per_class(predictions, test, 0)
recall_per_class(predictions, test, 1)             

class Mult_Nom_Gaus:
    
    def __init__(self, filename, split_ratio):
        self.filename = filename
        self.split_ratio = split_ratio
        self.operator = operator
        
    def data_transformer(self):
        self.lines = csv.reader(open(self.filename))
        self.dataset = list(self.lines)
        for i in range(len(self.dataset)):
            self.dataset[i] = [float(x) for x in self.dataset[i]]
        return self.dataset
    
    def train_test_split(self):
        self.test_number = round(len(self.dataset)*self.split_ratio)
        self.indexes = [i for i in range(0, len(self.dataset))]
        
        self.test_ind = random.choices(self.indexes, k = self.test_number)
        self.train_ind = [i for i in self.indexes if i not in self.test_ind]
        
        self.test_set = [self.dataset[i] for i in self.test_ind]
        self.train_set = [self.dataset[i] for i in self.train_ind]

        return self.train_set, self.test_set
   
    def separateByClass(self):
        self.separated = {}
        for i in range(len(self.train_set)):
            vector = self.train_set[i]
            if (vector[-1] not in self.separated):
                self.separated[vector[-1]] = []
            self.separated[vector[-1]].append(vector)
        return self.separated
    
    def mean(self, numbers):
        return sum(numbers)/float(len(numbers))

    def stdev(self, numbers):
        avg = mean(numbers)
        st_dev = math.sqrt(sum([(i - avg)**2 for i in numbers])/float(len(numbers)-1))
        return st_dev
    
    def summarize(self):
        self.summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*self.train_set)]
        del self.summaries[-1]
        return self.summaries
    
    def summarizeByClass(self):
        
        separated = separateByClass(self.train_set)
        self.summaries = {}
        for classValue, instances in separated.items():
            print(classValue, instances)
            self.summaries[classValue] = summarize(instances)
        return self.summaries
    
    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-((x-mean)**2/(2*stdev**2)))
        prob = (1/(math.sqrt(2*math.pi)*stdev))*exponent
        return prob
       
    def calculateClassProbabilities(self):
        
        self.summaries = summarizeByClass(self.train_set)
        self.probabilities = {}
        for classValue, classSummaries in self.summaries.items():
            self.probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = self.test_set[i]
                self.probabilities[classValue] *= calculateProbability(x, mean, stdev)
          
        return self.probabilities
    
    def predict(self, inputVector):
        self.probabilities = calculateClassProbabilities(self.summaries, inputVector) 
        
        return max(self.items(), key=self.operator.itemgetter(1))[0]

    def getPredictions(self):
        self.summaries = summarizeByClass(self.train_set)
        self.predictions = []
        for i in range(len(self.test_set)):
            result = predict(self.summaries, self.test_set[i])
            self.predictions.append(result)
        return self.predictions
    
    def get_accuracy(self):
    
        correct = []
        
        for i in range(len(self.test_set)):
            if self.test_set[i][-1] == self.predictions[i]:
                correct.append(1)
        else:
            correct.append(0)
        self.accuracy = sum(correct)/float(len(self.test_set))*100
        print('The accuracy is: ', self.accuracy)    
        return self.accuracy
    
    def precision_per_class(self, class_):
        
        correct = []
        class_positive = []
        
        for i in range(len(self.test_set)):
            
            if (self.test_set[i][-1] == class_) and (self.predictions[i] == class_):
                correct.append(class_)
                class_positive.append(class_)
        
            elif (self.test_set[i][-1] == class_) and (self.predictions[i] != class_):
                class_positive.append(class_)
        self.precision = round(len(correct)/len(class_positive), 3)
        print('The precision of the class ', class_, ' is: ', self.precision)
        return self.precision
    
    def recall_per_class(self, class_):
        
        correct = []
        class_predictions = []
        
        for i in range(len(self.test_set)):
            
            if (self.predictions[i] == class_) and (self.test_set[i][-1] == class_):
                correct.append(class_)
                class_predictions.append(class_)
        
            elif (self.predictions[i] == class_) and (self.test_set[i][-1] != class_):
                class_predictions.append(class_)
        self.recall = round(len(correct)/len(class_predictions), 3)
        print('The recall of the class ', class_, ' is: ', self.recall)
        return self.recall

our_lines = Mult_Nom_Gaus('data/pima-indians-diabetes.csv', 0.2)
our_lines.data_transformer()
train, test = our_lines.train_test_split()
our_lines.separateByClass()
our_lines.summarize()
our_lines.summarizeByClass()
#our_lines.calculateClassProbabilities()
pr = our_lines.getPredictions()
our_lines.get_accuracy()
our_lines.precision_per_class(1)
our_lines.precision_per_class(0)
our_lines.recall_per_class(1)
our_lines.recall_per_class(0)

train_ = pd.DataFrame(train)
test_ = pd.DataFrame(test)

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


model = GaussianNB()
model.fit(train_[train_.columns[0:8]], train_[train_.columns[-1]])
predicted = model.predict(test_[test_.columns[0:8]])
expected = test_[test_.columns[-1]]
probass = model.predict_proba(test_[test_.columns[0:8]])
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print(metrics.accuracy_score(expected, predicted))
