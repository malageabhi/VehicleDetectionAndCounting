# -*- coding: utf-8 -*-

import cv2


class ImageProcess(object):
    def __init__(self, image, height, width, subtractor):
        self.image = image
        self.height  = height
        self.width = width
        self.backgroundSubtractor = subtractor
        self.divider_forward = int(height/2) - 10
        self.divider_reverse = int(height/2) + 50
        
    def __get_center(self, x, y, w, h):
        '''
            calculate center of contour
        '''
        x1 = int(w / 2)
        y1 = int(h / 2)
    
        cx = x + x1
        cy = y + y1
        return (cx, cy)
    
    def __countVehicles(self, centroid,contourArea, extentContour):
        '''
            decision to count contour as car or bike in forward or reverse direction
        '''
        carF = 0
        carR = 0
        bikeF = 0
        bikeR = 0
        if abs(centroid[1] - self.divider_forward) == 0  and centroid[1] < self.divider_reverse:
            if contourArea < 1800 and extentContour < 0.5:
                bikeF =1
            else:
                carF = 1
        if abs(centroid[1] - self.divider_reverse) == 0 and centroid[1] > self.divider_forward:
            if contourArea < 1800 and extentContour < 0.5:
                bikeR =1
            else:
                carR = 1 

        return carF, bikeF, carR, bikeR

    
    
    def FilterMask(self,mask):
        '''
            morphological operations to fill holes in detetcted contours
            Input : backgraound subtracted image
            Output: image with holes filled
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # Fill any small holes
        closeImg = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        openImg = cv2.morphologyEx(closeImg, cv2.MORPH_OPEN, kernel)
        # Dilate to merge adjacent blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilationImage = cv2.dilate(openImg, kernel, iterations = 2)
      
        return dilationImage
    
    
        
    def DetectContours(self,mask):
        '''
            contour detection in from mask image
        '''
        matches = []
        carForward = 0
        carReverse = 0
        bikeForward = 0
        bikeReverse = 0
        MIN_CONTOUR_WIDTH = 50
        MIN_CONTOUR_HEIGHT = 50
    
        # Find the contours of any vehicles in the image
        contours,_= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            
            y = y  + int(self.height/2)-100
            contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT) 
          
            if not contour_valid:
                continue

            centroid = self.__get_center(x, y, w, h)
            contourArea = cv2.contourArea(contour)
            boundingBoxarea = w * h
            extentContour = contourArea / boundingBoxarea
            carF,bikeF,carR,bikeR = self.__countVehicles(centroid,contourArea, extentContour)
            carForward += carF
            bikeForward += bikeF
            carReverse += carR
            bikeReverse += bikeR

            matches.append(((x, y, w, h), centroid))
        
        return matches,(carForward,bikeForward,carReverse,bikeReverse)
        
    def process(self):
        '''
            process input image to detect and count number of cars and vehicles
        '''
        grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        croppedImage = grayImage[int(self.height/2)-100:int(self.height/2)+100, :]
        mask = self.backgroundSubtractor.apply(croppedImage)
        mask = self.FilterMask( mask)
        matches = self.DetectContours(mask)
        return matches

