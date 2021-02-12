#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:51:32 2020

@author: mhbrt
"""
import cv2


originalImage = cv2.imread('./temp/shell/test_squarerrr1.png')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
ret_img, bw_image = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

count = 0
for x in range(len(grayImage)):
    for y in range(len(grayImage)):
        if grayImage[x,y] == 0:
            count += 1