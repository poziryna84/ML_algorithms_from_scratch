# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:22:51 2020

@author: pozir
"""
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
    
class Mult_NB:
    
    def __init__(self):
        super().__init__
        
    def train_test_split(self, filename, text_column, label_column, split_ratio):
        
        self.filename = filename
        self.split_ratio = split_ratio
        self.text_column = text_column
        self.label_column = label_column
        data = pd.read_csv(self.filename)
        data = data[[self.text_column, self.label_column]]
        test_number = round(len(data)*self.split_ratio)
        indexes = [i for i in range(0, len(data))]
    
        test_ind = random.choices(indexes, k = test_number)
        train_ind = [i for i in indexes if i not in test_ind]
    
        self.test = data[data.index.isin(test_ind)]
        self.train = data[data.index.isin(train_ind)]
    
        return self.train, self.test
    
    def text_prepro(self, text):
        
        text = text.replace('@AmericanAir', '')
        text = ' '.join(['username' if i.startswith('@') else i for i in text.split()])
        text = ' '.join(['linkname' if i.startswith('http://') else i for i in text.split()])
        text = text.strip().lower()
        text = ' '.join([i for i in text.split() if i not in stopwords.words('english')])
        regex = re.compile('[^a-zA-Z]')
        text = regex.sub(' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip().lower()
        lemmatizer = WordNetLemmatizer()
        lem_text = ' '.join([lemmatizer.lemmatize(i) for i in text.split() if len(i) > 2])
        lem_text = lem_text.strip()
    
        return lem_text
       
    def addToBow(self, example, dict_index):
        
        if isinstance(example, np.ndarray):
            example = example[0]
            for token_word in example.split():
                self.bow_dicts[dict_index][token_word]+=1
                
    def training(self):
        
        self.classes = self.train[self.label_column].unique()
        self.bow_dicts = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
        self.examples = self.train[self.text_column]
        self.labels = self.train[self.label_column]
        
        if not isinstance(self.examples, np.ndarray):
            self.examples = np.array(self.examples)
        
    
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)
        
        for cat_index, cat in enumerate(self.classes):
            
            
            cat_examples = self.examples[self.labels == cat]
            
            clean_examples = [self.text_prepro(cat_example) for cat_example in cat_examples]
            
        
            clean_examples = pd.DataFrame(data = clean_examples)
            
        
            np.apply_along_axis(self.addToBow, 1, clean_examples, cat_index)
        
        # The list of the probabilities of each classes    
        prob_classes = np.empty(self.classes.shape[0])
        # The list of all the words from the data set, repeatitions included
        all_words = []
        # The number of words per every category:
        cat_word_counts = np.empty(self.classes.shape[0])
        
        for cat_index, cat in enumerate(self.classes):
            prob_classes[cat_index] = np.sum(self.labels == cat)/float(self.labels.shape[0]) 
            print('The probabilty of the ', cat, ' comments', ' is ', round(prob_classes[cat_index], 2))
            count = list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index] = np.sum(count)+1
            print('The total # of  words in ', cat, ' comments', ' is ', cat_word_counts[cat_index])
            all_words += self.bow_dicts[cat_index].keys()
        print('The total # of words', ' is ', len(all_words))    
        # The list of the unique words from the data set
        self.vocab = np.unique(np.array(all_words))
        # The number of the unique words:
        self.vocab_length = self.vocab.shape[0]
        print('The total # of unique words', ' is ', self.vocab_length) 
        self.denoms = np.array([cat_word_counts[cat_index] + self.vocab_length + 1 for cat_index,cat in enumerate(self.classes)])   
        self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], self.denoms[cat_index]) for cat_index, cat in enumerate(self.classes)]                               
        self.cats_info = np.array(self.cats_info)
        
       
    def getExampleProb(self, test_example):
        
        likelihood_prob = np.zeros(self.classes.shape[0]) #to store probability w.r.t each class
        for cat_index, cat in enumerate(self.classes):
            for test_token in test_example.split():
                test_token_counts = self.cats_info[cat_index][0].get(test_token, 0)+1
                # likelihood of this token word in whithin the given category 
                test_token_prob = test_token_counts/float(self.cats_info[cat_index][2]) 
                # to avoid underflow I am taking log
                likelihood_prob[cat_index] += np.log(test_token_prob)
            
        post_prob = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])   

                                 
        return post_prob
    
    def get_predictions(self):
        
        predictions = []
        for example in self.test[self.text_column]:
            clean_example = self.text_prepro(example)
            post_prob = self.getExampleProb(clean_example) #get prob of this example for each class        
            #pick the max value and map against self.classes
            predictions.append(self.classes[np.argmax(post_prob)])
        self.predictions = np.array(predictions)
                
        return  self.predictions
 
        
    def get_accuracy(self):
        
        correct = []
        true_label = self.test[self.label_column]

        if not isinstance(true_label, np.ndarray):
            true_label = np.array(true_label)
            for i in range(len(true_label)):

                if true_label[i] == self.predictions[i]:
                    correct.append(1)
                    
                else:
                    correct.append(0)
                    
            self.accuracy = sum(correct)/float(len(true_label))*100
            print('The accuracy is: ', self.accuracy)    
        return self.accuracy


mlt_nb = Mult_NB()
train, test = mlt_nb.train_test_split('data/Tweets.csv', 'text', 'airline_sentiment', 0.2,)
mlt_nb.training()
preds = mlt_nb.get_predictions()
mlt_nb.get_accuracy()

# Compare the accuracy of our model to the sklearn one:

def text_pre(text):
    
    text = text.replace('@AmericanAir', '')
    text = ' '.join(['username' if i.startswith('@') else i for i in text.split()])
    text = ' '.join(['linkname' if i.startswith('http://') else i for i in text.split()])
    text = text.strip().lower()
    text = ' '.join([i for i in text.split() if i not in stopwords.words('english')])
    regex = re.compile('[^a-zA-Z]')
    text = regex.sub(' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip().lower()
    lemmatizer = WordNetLemmatizer()
    lem_text = ' '.join([lemmatizer.lemmatize(i) for i in text.split() if len(i) > 2])
    lem_text = lem_text.strip()
    
    return lem_text
X_train = train['text'].apply(text_pre)
X_test = test['text'].apply(text_pre)

y_train = train['airline_sentiment']
y_test = test['airline_sentiment']

vectorizer = CountVectorizer()

vectors_train = vectorizer.fit_transform(X_train)
vectors_test = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(vectors_train, y_train)
prediction_clf = clf.predict(vectors_test)
print(accuracy_score(y_test, prediction_clf))

# Our implimentation: 75.42857142857143 accuracy 
# Sklearn implimentation: 0.7598776567567895 accuaracy.
