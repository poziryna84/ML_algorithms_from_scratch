# ML_algorithms_from_scratch
Coding ML algorithms from scratch.

Naive_bayes_Gaus.py: the implementation of Gaussian Naive Bayes 

In data folder there is pima-indians-diabetes.csv dataset comprising of 768 observations of women aged 21 and older. The dataset describes 
instantaneous measurements taken from patients, like age, blood workup, and the number of times they've been pregnant. Each record has a 
class value that indicates whether the patient suffered an onset of diabetes within 5 years. The values are 1 for Diabetic and 0 for 
Non-Diabetic.

To understand how Gaussian Naive Bayes works I am going to code it from scratch. I've broken the whole process down into the following steps: 1) handle data; 2) summarize data; 3) make predictions; 4) evaluate accuracy; 5) compare my implementation with sklearn one.

Naive_bayes_Gaus.py: the implementation of Multinomial Naive Bayes

In the data folder there is Tweets.csv dataset containing 14640 positive, neutral, or negative tweets for six US airlines. The comments and the sentiment labels ('positive', 'neutral', 'negative') are found in the 'text' and 'airline_sentiment' columns respectively.
The code represents step by step implementation of the Multinomial NB in python which is able to train dataset with different number of classes. The text preprocessing part is problem specific and better be modified depending on the data.
