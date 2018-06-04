#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 15:34:28 2017

@author: eli
"""

class BaseDataset:
    
    def __init__(self):
        pass
    
    def get_training_samples(self, num_samples):
        raise NotImplementedError()
        # return X, y
        
    def get_test_samples(self, num_samples):
        raise NotImplementedError()
        # return X, y
        
