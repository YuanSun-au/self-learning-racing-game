#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:54:21 2019

@author: carllindstrom
"""
import numpy as np
from classes import car
from NN_functions import population_to_weight_list, setup_network
from functions import wall_distance, crash_detection, fitness, take_action, draw_vision, write_text, check_pause
import pygame

#%%
def initialize_population(param_list, population_size, weight_range):
    
    n_params = sum(param_list)
    population = []
    max_range = weight_range[1]
    min_range = weight_range[0]
    
    for index in range(population_size):
        chromosome = np.random.rand(1, n_params)*(max_range - min_range) + min_range
        population.append(chromosome)
        
    
    return population
 
#%% 
def decode_population(population, cars, models, track, car_image, win,
                      input_size, output_size, layer_dimensions, param_list):
    
    # Set weights in network
    weight_list = population_to_weight_list(population, input_size, output_size, layer_dimensions, param_list)
    
    if cars == []:
        for car_index in range(len(population)):
            cars.append(car(x=track.start[0],y=track.start[1], car_image = car_image))
            models.append(setup_network(input_size, output_size, layer_dimensions))
            models[car_index].set_weights(weight_list[car_index])
            cars[car_index].draw_model(win)
    else:
        for index, icar in enumerate(cars):
            icar.__init__(track.start[0], track.start[1], car_image)
            icar.draw_model(win)
            models[index].set_weights(weight_list[index])
            
#%% 
def evaluate_population(cars, models, track, win, dt, draw_vision_lines, generation_number):
    
    pygame.display.update()
    
    pause = False
    running = True
    while running:
        #pygame.time.delay(dt)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                running = False
    
            if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                draw_vision_lines = not draw_vision_lines
        
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pause = True
                
            if event.type is pygame.QUIT: 
                pygame.quit()
                quit()
                
        
        pause = check_pause(pause, win)
                
                
        # Draw track
        win.blit(track.surface,(0,0))
    
        for model, icar in zip(models, cars):
        
            if icar.running:
                
                # Update car position and vision
                icar.vision()
                vision_lines, intersection_points, distance = wall_distance(icar, track)
                distance = distance.reshape((1,5))
                
                # Propagate through network
                nn_output = model.predict(distance)
                take_action(nn_output, icar)
            
                icar.update(dt=dt)
                
                crash_detection(icar, track)
                fitness(icar, distance, dt)
                
                draw_vision(draw_vision_lines, win, vision_lines, intersection_points)
                
                
            icar.draw_model(win)
               
        running_list = [icar.running for icar in cars]
        if not any(running_list):
            running = False
    
        
        write_text(win, generation_number)            
        pygame.display.update()
        
    fitness_list = [icar.fitness for icar in cars]
    
    return fitness_list, draw_vision_lines

#%%
def tournament_select(fitness_list, p_tournament, tournament_size):
    
    chosen_individuals = np.random.randint(0, len(fitness_list), size=tournament_size)
    temp_fitness = np.array(fitness_list)[chosen_individuals]
    
    r = np.random.uniform()

    
    while r > p_tournament and len(np.argwhere(temp_fitness != 0)) > 1:
        max_index = np.argmax(temp_fitness)
        temp_fitness[max_index] = 0
        r = np.random.uniform()
        
    max_index = np.argmax(temp_fitness)
    selected = chosen_individuals[max_index]
    return selected

#%%
def mutate(chromosome, p_mutation, weight_range):
    
    min_range = weight_range[0]
    max_range = weight_range[1]
    n_genes = len(chromosome[0])
    
    if p_mutation is None:
        p_mutation = 1.0/n_genes
    
    temp_count = 0
    for idx, gene in enumerate(chromosome[0]):
        r = np.random.uniform()
        if r < p_mutation:
            q = np.random.uniform()
            gene = q*(max_range - min_range) + min_range
            chromosome[0][idx] = gene
        temp_count += 1
    
    return chromosome
    
    
    
#%%
def crossover(chromosome1, chromosome2):
    
    n_genes = len(chromosome1[0])
    crossoverPoint = int(np.random.uniform()*(n_genes-1));
    
    newChromosome1 = np.zeros_like(chromosome1)
    newChromosome2 = np.zeros_like(chromosome2)
    
    for j in range(n_genes):
        if j <= crossoverPoint:
            newChromosome1[0,j] = chromosome1[0,j];
            newChromosome2[0,j] = chromosome2[0,j];
        else:
            newChromosome1[0,j] = chromosome2[0,j];
            newChromosome2[0,j] = chromosome1[0,j];

    return(newChromosome1,newChromosome2)
    
    
    
    
    
    