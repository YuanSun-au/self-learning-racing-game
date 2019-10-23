#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:30:14 2019

@author: carllindstrom
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

def setup_network(input_size, output_size, layer_dimensions):
    
    model = Sequential()
    
    model.add(Dense(layer_dimensions[0], input_dim=input_size))
    model.add(Activation('relu'))
    
    for layer_dim in layer_dimensions[1:]:
        model.add(Dense(layer_dim))
        model.add(Activation('relu'))
        
   # model.add(Dense(output_size))
   # model.add(Activation('softmax'))
    
    model.add(Dense(output_size))
    if output_size == 3 or output_size == 2 :    
        model.add(Activation('softmax'))
    else:
        model.add(Activation('tanh'))
    
    return model
    
#%%

def calculate_parameters(input_size, output_size, layer_dimensions):
    
    n_params = []
    n_params.append( (input_size+1)*layer_dimensions[0] )
    
    for index, layer_dim in enumerate(layer_dimensions[1:]):
        n_params.append( (layer_dimensions[index] + 1)*layer_dim )
        
    n_params.append( (layer_dimensions[-1] + 1)*output_size )
    
    return n_params

#%%
    
def population_to_weight_list(population, input_size, output_size, layer_dimensions, param_list):
    
    weight_list = [[] for x in range(len(population))]
    param_list = np.cumsum(param_list)
    
    for chromosome_index, chromosome in enumerate(population):
        first_layer_weights = chromosome[0, 0:param_list[0]]        
        first_layer = np.zeros((input_size, layer_dimensions[0]))

        for index, row in enumerate(first_layer):
            first_layer[index,:] = first_layer_weights[index*layer_dimensions[0] : (index+1)*layer_dimensions[0]]
        
        weight_list[chromosome_index].append(first_layer)
        activation_weights = np.zeros(( layer_dimensions[0]))
        weight_list[chromosome_index].append(activation_weights)
        
        
        for prev_index, layer_dim in enumerate(layer_dimensions[1:]):
            input_dim = layer_dimensions[prev_index]
            output_dim = layer_dim
            
            layer_weights = chromosome[ 0, param_list[prev_index]:param_list[prev_index+1] ]
            #print('First index: ', param_list[prev_index])
            #print('Second index: ', param_list[prev_index+1])
            #print(layer_weights)
            layer = np.zeros((input_dim, output_dim))
            
            for index, row in enumerate(layer):
                layer[index,:] = layer_weights[index*output_dim : (index+1)*output_dim]
            
            weight_list[chromosome_index].append(layer)
            activation_weights = np.zeros(( output_dim))
            weight_list[chromosome_index].append(activation_weights)
            
        last_layer_weights = chromosome[0, param_list[-2]:param_list[-1]]        
        last_layer = np.zeros((layer_dimensions[-1], output_size))

        for index, row in enumerate(last_layer):
            last_layer[index,:] = last_layer_weights[index*output_size : (index+1)*output_size]
        
        weight_list[chromosome_index].append(last_layer)
        activation_weights = np.zeros(( output_size))
        weight_list[chromosome_index].append(activation_weights)
    
            
    return weight_list
        