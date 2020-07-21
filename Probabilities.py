# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 07:47:52 2020

@author: pozir
"""

def binom_pmf(suc_proba, k_suc, n_trials):
    '''Function that returns the probability of k 
    successes in n trials given the constant probability of
    suc_proba of k_suc.
   '''                                
    fail_proba = 1 - suc_proba
    n_fail = n_trials - k_suc
    x = n_trials - k_suc + 1
    numerator = 1
    denominator = 1
    for i in range(n_trials+1)[x:]:

        numerator *= i
    for i in range(k_suc+1):
        if i > 0:
            denominator*=i
    comb = numerator/denominator
    proba = comb *(suc_proba**k_suc)*(fail_proba**(n_fail))
            
    return proba

binom_pmf(0.2, 4, 1)

def binom_cdf(suc_proba, cond,  k_suc, n_trials):
    '''Function that returns the cumulative probability 
    of more than or less or equal to (depending on the 'cond' specified) k 
    successes in n trials given the constant probability of suc_proba of k_suc.
    cond: takes string values of either '>' or '<= '.
   '''                              
    cum_prob = 0
    for i in range(k_suc + 1):
        if i > 0:
            cum_prob += binom_pmf(suc_proba, n_trials, i)
    if cond == '>':
        cum_prob = 1 - cum_prob
        
    
    return cum_prob
            
binom_cdf(0.9, '>', 1, 3)   

def geom_pdf(proba, n): 
    '''Function that returns the probability of the event with its probability
    iqual to proba happening on the nth trial. 
    '''
    fail_proba = 1 - proba
    return (fail_proba**(n-1))*proba

geom_pdf(0.05, 4)

def geom_cdf(proba, cond, n):
    '''Function that returns the probability that it will take the event, with
    the probability proba, n or less times or  more than n times (depending
    on the condition 'cond' specified) to happen.
    cond: takes string values of either '>' or '<= '
    '''
    fail_proba = 1 - proba
    cum_prob = proba
    for i in range(n):
        if i > 0:
            aux_proba = (fail_proba**i)*proba
            
            cum_prob+= aux_proba
            print(cum_prob)
    if cond == '>':
        cum_prob = 1 - cum_prob
        
            
    return cum_prob
            
geom_cdf(0.02, '<=', 3)            


