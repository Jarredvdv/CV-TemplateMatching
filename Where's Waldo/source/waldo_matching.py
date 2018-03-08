#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:40:30 2018

@author: jarred
"""

# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Insert the location of the images
puzzle_address = 'Convention.jpg'
waldo_address = 'Wally.png'

# load the puzzle and waldo images
puzzle = cv2.imread(puzzle_address)
waldo = cv2.imread(waldo_address)
(waldoHeight, waldoWidth) = waldo.shape[:2]

# Display the dimensions and plot the image of Waldo
print("Height of Template: %d & Width of Template: %d" %(waldoHeight, waldoWidth))

# cv2 reads the image in BGR, we need to convert it to RGB values to see the plot. 
# What happens if you do not convert?
# Go ahead and try it out. Comment the next line and run the cell.
waldo_rgb = cv2.cvtColor(waldo,cv2.COLOR_RGB2BGR)

plt.figure(figsize=(1,1))
plt.imshow(waldo_rgb)

plt.figure(figsize=(15,15))
puzzle_rgb = cv2.cvtColor(puzzle,cv2.COLOR_RGB2BGR)

plt.imshow(puzzle_rgb)

# find the waldo in the puzzle
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)

(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

# grab the bounding box of waldo and extract him from
# the puzzle image
topLeft = maxLoc
botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

# construct a darkened transparent 'layer' to darken everything
# in the puzzle except for waldo
mask = np.zeros(puzzle.shape, dtype = "uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
 
# display the images
cv2.imwrite("Puzzle_Result.jpg", puzzle)
result_rgb = cv2.cvtColor(puzzle,cv2.COLOR_RGB2BGR)
plt.figure(figsize=(15,15))
plt.imshow(result_rgb)

plt.figure(figsize = (3,3))
plt.imshow(cv2.cvtColor(cv2.imread('waldo_books_dim.jpg'), cv2.COLOR_BGR2RGB))

plt.figure(figsize = (15,15))
plt.imshow(cv2.cvtColor(cv2.imread('waldo_zoo.jpg'), cv2.COLOR_BGR2RGB))