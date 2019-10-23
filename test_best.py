#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:16:28 2019

@author: Johan
"""

from classes import track
import pygame
import numpy as np
from NN_functions import calculate_parameters
from GA_functions import (decode_population, evaluate_population)
from functions import write_text

#%% Load the best individual
best_individual = np.load('best_individual.npy')
print('Best individual: ', best_individual)

#%% Parameters
dt = 10
bg_color = (158,160,163)    # RGB
line_color = (0,0,0)
white = (255, 255, 255) 
green = (0, 255, 0) 
blue = (0, 0, 128)
draw_vision_lines = False
save_best = False


# Network params
input_size = 5
output_size = 1
layer_dimensions = [5]

#%%
pygame.init()
win = pygame.display.set_mode((1280,720))
pygame.display.set_caption('CJ Racing Game')
car_image = pygame.image.load('car40_20.png')

#Load track
track = track('test_track.csv')
background_image = pygame.image.load("track_bg.png").convert()
track.draw(line_color,bg_color, background_image)

# Initialize population
param_list = calculate_parameters(input_size, output_size, layer_dimensions)
population = [best_individual]

cars = []
models = []

pygame.event.pump()

win.blit(track.surface,(0,0))
write_text(win, None)

decode_population(population, cars, models, track, car_image, win,
                  input_size, output_size, layer_dimensions, param_list)

fitness_list, draw_vision_lines = evaluate_population(cars, models, track, 
                              win, dt, draw_vision_lines, None)


pygame.quit()