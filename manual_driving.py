# -*- coding: utf-8 -*-

#Imports
from classes import car, track
from functions import (crash_detection, wall_distance, fitness, draw_vision, 
                       take_action, check_pause, write_text)
import pygame
import numpy as np
from pygame.locals import*
import math
from NN_functions import setup_network, calculate_parameters, population_to_weight_list
from GA_functions import (initialize_population, decode_population, evaluate_population,
                          tournament_select, mutate)
import copy

#%% Parameters
dt = 10
bg_color = (158,160,163)    # RGB
line_color = (0,0,0)
white = (255, 255, 255) 
green = (0, 255, 0) 
blue = (0, 0, 128)

#%%
pygame.init()
win = pygame.display.set_mode((1280,720))
pygame.display.set_caption('CJ Racing Game')
car_image = pygame.image.load('car40_20.png')

#Load track
track = track('raceTrackForBG.csv')
background_image = pygame.image.load("track_bg.png").convert()
track.draw(line_color,bg_color, background_image)

#Create car object
icar = car(x=track.start[0],y=track.start[1], car_image = car_image)

#Crash text
font = pygame.font.Font('freesansbold.ttf', 32)
text = font.render('CRASH!', True, green, blue)
textRect = text.get_rect()
textRect.center = (1080 // 2, 720 // 2) 

draw_vision_lines = True
running = True
pause = False
while running:
    #pygame.time.delay(dt)
    
    win.blit(track.surface,(0,0))
    write_text(win, 0)
    
    for event in pygame.event.get():
            
        if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
            draw_vision_lines = not draw_vision_lines
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            pause = True
        
        if event.type is pygame.QUIT: 
            pygame.quit()
            quit()
    
    keyes = pygame.key.get_pressed()
    if keyes[pygame.K_LEFT]:
        icar.steering_angle -= icar.steering_rate
    elif keyes[pygame.K_RIGHT]:
        icar.steering_angle += icar.steering_rate
    else:
        icar.steering_angle = 0

    icar.vision()
    vision_lines, intersection_points, distance = wall_distance(icar, track)
    icar.update(dt=dt)
    
    crash_detection(icar, track)
    fitness(icar, distance, dt)
    
    draw_vision(draw_vision_lines, win, vision_lines, intersection_points)
    icar.draw_model(win)
    
    if not icar.running:
        running = False
    
    pause = check_pause(pause, win)
    pygame.display.update()
    
        
pygame.quit()