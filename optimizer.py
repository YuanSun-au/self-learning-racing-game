# -*- coding: utf-8 -*-

#Imports
from classes import track
import pygame
import numpy as np
from NN_functions import calculate_parameters
from GA_functions import (initialize_population, decode_population, evaluate_population,
                          tournament_select, mutate, crossover)
from functions import write_text, save_individual
import copy

#%% Parameters
dt = 10
bg_color = (158,160,163)    # RGB
line_color = (0,0,0)
white = (255, 255, 255) 
green = (0, 255, 0) 
blue = (0, 0, 128)
draw_vision_lines = False
save_best = True

# GA params
n_generations = 60
population_size = 40
initial_weight_range = (-1,1)
weight_range = (-15,15)
p_tournament = 0.75
tournament_size = 10
p_crossover = 0.8
n_copies = 3

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
track = track('training_track.csv') #track('raceTrackForBG.csv')
background_image = pygame.image.load("track_bg.png").convert()
track.draw(line_color,bg_color, None)#background_image)

# Initialize population
param_list = calculate_parameters(input_size, output_size, layer_dimensions)
population = initialize_population(param_list, population_size, initial_weight_range)

cars = []
models = []

for generation in range(n_generations):

    pygame.event.pump()
    
    win.blit(track.surface,(0,0))
    write_text(win, generation+1)
    
    decode_population(population, cars, models, track, car_image, win,
                      input_size, output_size, layer_dimensions, param_list)

    fitness_list, draw_vision_lines = evaluate_population(cars, models, track, 
                                  win, dt, draw_vision_lines, generation+1)
    
    best_individual_index = np.argmax(fitness_list)
    best_individual = copy.deepcopy(population[best_individual_index])
    
    temp_population = copy.deepcopy(population)
    for i in range(population_size, 2):
        selected1 = tournament_select(fitness_list, p_tournament, tournament_size)
        selected2 = tournament_select(fitness_list, p_tournament, tournament_size)
        chromosome1 = copy.deepcopy(population[selected1])
        chromosome2 = copy.deepcopy(population[selected2])
        
        r = np.random.uniform()
        if r < p_crossover:
            new_chromosome1, new_chromosome2 = crossover(chromosome1, chromosome2)
            temp_population[i] = copy.deepcopy(new_chromosome1)
            temp_population[i+1] = copy.deepcopy(new_chromosome2)
        else:
            temp_population[i] = population[selected1]
            temp_population[i+1] = population[selected2]
    
    for j in range(population_size):
        chromosome = copy.deepcopy(temp_population[j])
        chromosome = mutate(chromosome, p_mutation = None, weight_range=weight_range)
        temp_population[j] = copy.deepcopy(chromosome)
    
    for k in range(n_copies):
        temp_population[k] = copy.deepcopy(best_individual)
        
    population = copy.deepcopy(temp_population)
    save_individual('best_individual',best_individual,save_best)

pygame.quit()