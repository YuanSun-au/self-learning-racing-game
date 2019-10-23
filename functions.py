# -*- coding: utf-8 -*-
import numpy as np
import pygame 
import math

def line_intersection(line1, line2):
    p0_x = float(line1[0])
    p0_y = float(line1[1])
    
    p1_x = float(line1[2])
    p1_y = float(line1[3])
    
    p2_x = float(line2[0])
    p2_y = float(line2[1])
    
    p3_x = float(line2[2])
    p3_y = float(line2[3])
    
    s1_x = p1_x - p0_x    
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x    
    s2_y = p3_y - p2_y
    

    epsilon = 0.000001
    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y + epsilon)
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y + epsilon)

    if (s >= 0.0 and s <= 1.0 and t >= 0.0 and t <= 1.0):
        # Collision detected
        i_x = p0_x + (t * s1_x);
        i_y = p0_y + (t * s1_y);
        intersection = 1
    else:
        i_x = 0.0
        i_y = 0.0
        intersection = 0

    return (intersection, i_x, i_y)

#%%
def crash_detection(car, track):
    #Left and right side of the car
    sides = [car.corners[0] + car.corners[2],
             car.corners[1] + car.corners[3]]
    
    for side in sides:
        for line in track.coordinates[:-1]:
            intersection, x, y = line_intersection(side,line)
            if intersection == 1:
                car.running = False
                car.velocity = 0
                car.steering_rate = 0
                #win.blit(text, textRect)
            
    
#%%
def wall_distance(car, track):
    
    vision_lines = [(car.x,car.y) + point for point in car.vision_points]
    
    distance = np.ones((5,1))*car.vision_length
    intersection_points = car.vision_points
    
    for i, vision_line in enumerate(vision_lines):
        for line in track.coordinates[:-1]:
            intersection, x, y = line_intersection(vision_line,line)
            if intersection == 1:
                temp_distance = math.sqrt( (car.x - x)**2 + (car.y - y)**2 )
                if temp_distance < distance[i]:
                    distance[i] = temp_distance
                    intersection_points[i] = (int(x),int(y))
    
    return vision_lines, intersection_points, distance
    
#%%
    
def fitness(car, distance, dt):
    
    car.fitness += np.mean(distance) /car.vision_length/dt
    #car.fitness += (distance[0,2])**2 /car.vision_length/dt
    
    
#%%

def draw_vision(show, win, vision_lines, intersection_points):
    
    RED = (255, 0, 0)
    BLUE = (61, 122, 166)
    if show:
        pygame.draw.line(win, BLUE, (int(vision_lines[0][0]), int(vision_lines[0][1])),
                                           (int(vision_lines[0][2]), int(vision_lines[0][3])), 2)
        pygame.draw.line(win, BLUE, (int(vision_lines[1][0]), int(vision_lines[1][1])),
                                           (int(vision_lines[1][2]), int(vision_lines[1][3])), 2)
        pygame.draw.line(win, BLUE, (int(vision_lines[2][0]), int(vision_lines[2][1])),
                                           (int(vision_lines[2][2]), int(vision_lines[2][3])), 2)
        pygame.draw.line(win, BLUE, (int(vision_lines[3][0]), int(vision_lines[3][1])),
                                           (int(vision_lines[3][2]), int(vision_lines[3][3])), 2)
        pygame.draw.line(win, BLUE, (int(vision_lines[4][0]), int(vision_lines[4][1])),
                                           (int(vision_lines[4][2]), int(vision_lines[4][3])), 2)
        pygame.draw.circle(win, RED, (int(intersection_points[0][0]), int(intersection_points[0][1])), 5)
        pygame.draw.circle(win, RED, (int(intersection_points[1][0]), int(intersection_points[1][1])), 5)
        pygame.draw.circle(win, RED, (int(intersection_points[2][0]), int(intersection_points[2][1])), 5)
        pygame.draw.circle(win, RED, (int(intersection_points[3][0]), int(intersection_points[3][1])), 5)
        pygame.draw.circle(win, RED, (int(intersection_points[4][0]), int(intersection_points[4][1])), 5)
        
    
#%%
        
def take_action(nn_output, car):    
    
    number_of_outputs = len(nn_output[0])

    if number_of_outputs == 3:   
        action = np.argmax(nn_output)
        if action == 0:
            car.steering_angle -= car.steering_rate
        elif action == 2:
            car.steering_angle += car.steering_rate
        else:
            car.steering_angle = 0
            
    elif number_of_outputs == 2:
        if nn_output[0][0] > 0.8:
            car.steering_angle -= car.steering_rate
        elif nn_output[0][1] > 0.8: 
            car.steering_angle += car.steering_rate
        else:
            car.steering_angle = 0
    else:
        if nn_output < -0.5:
             car.steering_angle -= car.steering_rate
        elif nn_output > 0.5:
             car.steering_angle += car.steering_rate
        else:
             car.steering_angle = 0
    
#%%
def write_text(surface, generation_number):
    white = (255, 255, 255) 
    green = (0, 255, 0) 
    blue = (0, 0, 128)
    black = (0, 0, 0)
    bg_color = (128, 197, 106)
    font_style = 'rockwellttc'
    
    font_gen = pygame.font.SysFont(font_style, 18)
    font_info = pygame.font.SysFont(font_style, 10)
    if generation_number is None:
        gen_string = 'Test of the best'
    else:
        gen_string = 'Generation: ' + str(generation_number)
        
    info_string = 'Press "v" to toggle vision, "SPACE" to terminate generation and "p" to pause.'
    text_gen = font_gen.render(gen_string, True, black, bg_color)
    text_info = font_info.render(info_string, True, black, bg_color)
    text_gen_rect = text_gen.get_rect()
    text_info_rect = text_gen.get_rect()
    text_gen_rect.center = (100,30)
    text_info_rect.center = (900,30) 
    surface.blit(text_gen, text_gen_rect)
    surface.blit(text_info, text_info_rect)

#%%    
def check_pause(pause, surface):
    font_style = 'rockwellttc'
    red = (255, 0, 0)
    bg_color = (128, 197, 106)
    while pause:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pause = False
                
            if event.type is pygame.QUIT: 
                pygame.quit()
                quit()
            
        font = pygame.font.SysFont(font_style, 18)
        string = 'PAUSED'
        text = font.render(string, True, red, bg_color)
        rect = text.get_rect()
        rect.center = (1280/2, 30)
        surface.blit(text, rect)
        pygame.display.update()
    
    return pause
    
#%%
def save_individual(filename,best_individual, save_best):
    if save_best:
        np.save(filename, best_individual)
    
    
    
    
    
    
    
    