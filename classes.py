# -*- coding: utf-8 -*-
import numpy as np
import pygame
import math

class car:
    def __init__(self, x, y, car_image, size=(40,20)):
        self.x = x
        self.y = y
        self.size = size
        self.image = car_image
        self.steering_angle = 0
        self.steering_rate = 0.1
        self.vehicle_angle = -math.pi/2
        self.velocity = 1.8
        self.xVel = 0
        self.yVel = 0
        self.d = np.sqrt( (size[0]/2)**2 + (size[1]/2)**2 )
        self.vision_length = 400
        self.fitness = 0
        self.running = True
        
    def update(self,dt):
        
        if self.steering_angle > 0.25:
            self.steering_angle = 0.25
        elif self.steering_angle < -0.25:
            self.steering_angle = -0.25
        
        length = self.size[0]
        delta = self.steering_angle
        v = self.velocity
        theta = self.vehicle_angle
        
        lr = length/2.0    
        beta = np.arctan(lr*np.tan(delta)/length)
        theta_dot = v*np.cos(beta)*np.tan(delta)/length
        theta += theta_dot*dt
        theta = theta % (2.0*math.pi)
        
        self.xVel = np.cos(theta + beta)
        self.yVel = np.sin(theta + beta)
        self.vehicle_angle = theta
        
        self.x += self.xVel*dt
        self.y += self.yVel*dt
        
        width = self.size[1]
        angle = self.vehicle_angle
        alpha = np.tan( (width/2.0)/(length/2.0) )
        
        x_1 = float( self.x + np.cos(alpha - angle) * self.d )
        x_2 = float( self.x + np.cos(alpha + angle) * self.d )
        x_3 = float( self.x - np.cos(alpha + angle) * self.d )
        x_4 = float( self.x - np.cos(alpha - angle) * self.d )
        
        y_1 = float( self.y - np.sin(alpha - angle) * self.d )
        y_2 = float( self.y + np.sin(alpha + angle) * self.d )
        y_3 = float( self.y - np.sin(alpha + angle) * self.d )
        y_4 = float( self.y + np.sin(alpha - angle) * self.d )
        
        self.corners = [(x_1,y_1),(x_2,y_2),(x_3,y_3),(x_4,y_4)]
        
    def draw_model(self,win):
        angle = np.degrees(-self.vehicle_angle)
        img_rotated = pygame.transform.rotate(self.image, angle)
        rot_rect = img_rotated.get_rect(center=(self.x, self.y))
        win.blit(img_rotated, rot_rect)
      
    def vision(self):
        vision_length = self.vision_length
        x_1 = float(self.x + np.cos(math.pi/2.0 - self.vehicle_angle)*vision_length )
        x_2 = float(self.x + np.cos(math.pi/4.0 - self.vehicle_angle)*vision_length )
        x_3 = float(self.x + np.cos(self.vehicle_angle)*vision_length )
        x_4 = float(self.x + np.cos(math.pi/4.0 + self.vehicle_angle)*vision_length )
        x_5 = float(self.x + np.cos(math.pi/2.0 + self.vehicle_angle)*vision_length )
        
        y_1 = float(self.y - np.sin(math.pi/2.0 - self.vehicle_angle)*vision_length )
        y_2 = float(self.y - np.sin(math.pi/4.0 - self.vehicle_angle)*vision_length )
        y_3 = float(self.y + np.sin(self.vehicle_angle)*vision_length )
        y_4 = float(self.y + np.sin(math.pi/4.0 + self.vehicle_angle)*vision_length )
        y_5 = float(self.y + np.sin(math.pi/2.0 + self.vehicle_angle)*vision_length )
        
        self.vision_points = [(x_1,y_1), (x_2,y_2), (x_3,y_3), (x_4,y_4), (x_5,y_5)]
#%%
import csv
import pygame
from pygame.locals import*

class track:
    def __init__(self,filename):
        
        self.surface = pygame.Surface((1280,720))
        track_coordinates = []
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            for row in csv_reader:
                track_coordinates.append(list(map(int,row)))
            self.coordinates=track_coordinates   
            
            finish_line = track_coordinates[-1]
            x_start = (finish_line[0] + finish_line[2])/2
            y_start = (finish_line[1] + finish_line[3])/2
            self.start = (x_start,y_start)
                
    
    def draw(self,line_color,bg_color, background_image):
        
        win = self.surface
        if background_image is not None:
            win.blit(background_image, [0, 0])
        else:
            win.fill(bg_color)
    
            for line in self.coordinates[:-1]:
                pygame.draw.line(win, line_color, 
                                 (line[0], line[1]),
                                 (line[2], line[3]), 4)
            finish_line = self.coordinates[-1]    
            pygame.draw.line(win, (255,255,255),
                             (finish_line[0], finish_line[1]),
                             (finish_line[2], finish_line[3]), 1)