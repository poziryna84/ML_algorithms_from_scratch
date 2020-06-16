# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:39:14 2020

@author: pozir
"""
import pandas as pd
import numpy as np
#import requests
#import lxml.html as lh
#import scipy
from scipy.stats import chi2

class GodnessOfFit:
    def __init__(self):
        super().__init__
    def chi_square_two_tables(self, input):
        self.input = input
        if not isinstance(self.input, pd.DataFrame):
            self.input = pd.DataFrame(input)
        self.degree_freedom = (self.input.shape[0]-1)*(self.input.shape[1]-1)
        print('The degree of freedom is:')
        print(self.degree_freedom)
        sums_vert = list(self.input.sum(axis=0))
                    
        df = pd.concat([self.input,pd.DataFrame(sums_vert).T], axis = 0).reset_index(drop=True)
                     
        df['tot'] = df.sum(axis=1)
        total_sum = list(df['tot'])[-1]
        df['Rel_freq'] = df['tot']/total_sum
        self.chi_value = 0
        for row_ind in range(len(df)-1):
            prob = df['Rel_freq'][row_ind]
            for col_ind in range(df.shape[1]-2):
                act_value = df[col_ind][row_ind]
                exp_value = prob*(sums_vert[col_ind])          
                self.chi_value += ((act_value - exp_value)**2/exp_value)
        return self.chi_value
    
    def chi_square_one_table(self, input, probas):
        self.input = input
        self.degree_freedom = len(self.input) -1
        summ = sum(self.input)
        print(summ)
        self.chi_value = 0
        for i in range(len(self.input)):
            exp_value = summ * probas[i]
            print(exp_value)
            print(self.chi_value)
            self.chi_value += (((self.input[i] - exp_value)**2)/exp_value)
            
        return self.chi_value
    
    def p_value(self):
        
        self.pval = 1-chi2.cdf(self.chi_value, self.degree_freedom)
        return self.pval
                
gf = GodnessOfFit()
gf.chi_square_two_tables([[6, 78], [9, 55], [15, 133], [6, 58]])
gf.p_value()

gf.chi_square_one_table([1968, 14, 8, 10], [0.986, 0.005, 0.004, 0.005])
gf.p_value()

data = pd.DataFrame([[16,30, 14], [27,23,10], [29,16,15]])
gf.chi_square_two_tables(data)
gf.p_value()