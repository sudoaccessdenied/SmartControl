#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:42:32 2020

@author: anon
"""

import cv2 as cv
import numpy as np
import pyautogui as pag
import dlib
from imutils import face_utils
import imutils



def direction(nosePoint, center, w, h):
    nx, ny = nosePoint
    x, y = center

    if nx > x +  w:
        return 'right'
    elif nx < x - w:
        return 'left'

    if ny > y + h:
        return 'down'
    elif ny < y - h:
        return 'up'

    return 'IDLE'

def isWatching(nosePoint,center ,w,h):
    if np.linalg.norm(np.array(nosePoint)-np.array(center))<=w/2:
        return True
    
    return False



def main():
    cam = cv.VideoCapture(0)
    cam_w = 640
    cam_h = 480
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
    
    COLORS = {"cyan":(252, 202, 3),"yellow":(3, 252, 132),"red":(0,0,255)}
    CENTER = (0,0)
    RECTANGLE = (45,25)
    START = False
    
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    # count = 0
    pause =0
    
    isPaused = False
    while True:
        
        ret,frame = cam.read()
        
        frame = cv.flip(frame, 1)
        frame = imutils.resize(frame, width=cam_w, height=cam_h)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
        faces = detector(gray,0) 
        
        if len(faces)>0:
            face = faces[0]
        else:
            cv.imshow("Scroll",frame)
            key = cv.waitKey(1) & 0xFF
            continue
        
        landmarks = predictor(gray,face)
        shape = face_utils.shape_to_np(landmarks)
        nose = shape[nStart:nEnd]
    
        
        
        
        nosePoint = (nose[3, 0], nose[3, 1])
        if(START == False ):
            CENTER = nosePoint
            cv.putText(frame, "Scrolling OFF! Press S to Start", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['red'], 2)
            for (x,y) in shape:
                cv.circle(frame,(x,y),2,COLORS['yellow'],-1)
        elif START == True:
            
            cv.putText(frame, "Scrolling ON! Press S to Stop", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['red'], 2)
            x,y = CENTER
            nx,ny = nosePoint
            w ,h = RECTANGLE
            
            cv.rectangle(frame,(x-w,y-h),(x+w,y+h),COLORS["cyan"],2)
            cv.line(frame,CENTER,nosePoint,COLORS["red"],2)
            
            dir = direction(nosePoint, CENTER, w, h)
            cv.putText(frame, dir.upper(), (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['red'], 2)
            
            if isWatching(CENTER,nosePoint,w,h) and isPaused==True:
                    pag.press('k')
                    isPaused = False
            
            if dir == 'right':
                
                if  pause >3 and isPaused == False:
                    pag.press('k')
                    pause = 0
                    isPaused = True
                
                else:
                    pause+=1
                
                
            elif dir == 'left':
               
                if  pause >3 and isPaused == False:
                    pag.press('k')
                    pause = 0
                    isPaused = True
                
                else:
                    pause+=1
                    
            elif dir == 'up':
    
                    pag.scroll(1)
            elif dir == 'down':
                    pag.scroll(-1)
        
        
        
        cv.imshow("Scroll",frame)
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('s'):
            START= not START
        
        if key ==ord('q'):
            break;    
           
    
    cam.release()
    cv.destroyAllWindows()


if __name__ =="__main__":
    main()
