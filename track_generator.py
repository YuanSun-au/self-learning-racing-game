#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:27:08 2019

@author: carllindstrom
"""

import tkinter as tk
from tkinter import *
from PIL import ImageTk
import csv


def draw_line(event):

    if str(event.type) == 'ButtonPress':
        canvas.old_coords = event.x, event.y

    elif str(event.type) == 'ButtonRelease':
        x, y = event.x, event.y
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1, fill="red",width=3)
        
        # Save coordinates of lines
        row = [x1, y1, x, y]
        track_coordinates.append(row)
        

# Load GUI and canvas
root = tk.Tk()
canvas = tk.Canvas(root, width=1280, height=720)
canvas.pack(expand = YES, fill = BOTH)
image = ImageTk.PhotoImage(file = "track_bg.png")
#canvas.create_image(0, 0, image = image, anchor = NW) # Uncomment to include background image
canvas.old_coords = None
track_coordinates = []

# Draw lines between mouse click and release
root.bind('<ButtonPress-1>', draw_line)
root.bind('<ButtonRelease-1>', draw_line)
root.mainloop()

# Save line coordinates to .csv file
with open('new_track.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(track_coordinates)
            
csvFile.close() 