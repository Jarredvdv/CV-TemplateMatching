#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:45:22 2018

@author: jarred
"""

import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

puzzle = cv2.imread('waldo_zoo.jpg') #variable that stores array of puzzle
template = cv2.imread('waldo_books_dim.jpg')#variable stores array of wally
(waldoHeight, waldoWidth) = template.shape[:2]

found = None #we need a variable to compare correlations between scaled puzzles


for scale in np.linspace(1, 2, 20): #Loops through and supplies a scale for each image
    
    scaled_puzzle = imutils.resize(puzzle, width = int(puzzle.shape[1] * scale)) #Scales images according to provided scale        
    result = cv2.matchTemplate(scaled_puzzle, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    
    if found is None or maxVal > found[0]: #Checks if current scaled puzzle matches better than current best
        found = (maxVal, maxLoc) #if so, we need to store the resized puzzle & values
        winner = scaled_puzzle

# grab the bounding box of waldo and extract him from
# using the best matched puzzle size
maxLoc = found[1]
topLeft = maxLoc
botRight = (int((topLeft[0] + waldoWidth)), int((topLeft[1] + waldoHeight)))
roi = winner[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

# construct a darkened transparent 'layer' to darken everything
# in the puzzle except for waldo
mask = np.zeros(winner.shape, dtype = "uint8")
winner = cv2.addWeighted(winner, 0.25, mask, 0.75, 0)
winner[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
 
# display the images
cv2.imwrite("Puzzle_Result.jpg", winner)
result_rgb = cv2.cvtColor(winner,cv2.COLOR_RGB2BGR)
plt.figure(figsize=(15,15))
plt.imshow(result_rgb)