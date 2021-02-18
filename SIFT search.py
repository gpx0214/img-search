#!/usr/bin/env Python
# -*- coding:UTF-8 -*-
import numpy as np
import cv2
import os
import sys,getopt
from matplotlib import pyplot as plt
import time

#print(os.name)
#print(os.getcwd())
#print(os.path.split(os.path.realpath(__file__)))
#print(sys.argv[0])
#print(sys.path[0])
#os.system('cd /d '+os.path.split(os.path.realpath(__file__))[0])
sift = cv2.xfeatures2d.SIFT_create()
cnt_time=0
#sift = cv2.SIFT()
bf = cv2.BFMatcher()
start_time=time.time()
try:
    fn0, fn1 = args[1:]
except:
    fn0 = 'D:\\49de16d4naec011346dc3&690.jpg' #D:\\bluestacks_screen_shot170107-1.429coins50.jpg
    fn1 = 'D:\\IMG_20160124_144227_640_480.jpg' #D:\\bluestacks_screen_shot170107-1.png
path=os.path.split(os.path.realpath(__file__))[0]
print(path)
img0=cv2.imread(fn0, 0)
if img0 is None:
    print('img0 is None')
else:
    #print img0.shape #
    kp0, des0 = sift.detectAndCompute(img0,None)
    print('%4d features|%s|%s' % (len(kp0),img0.shape,fn0))

for filename in os.listdir(path):
    if( filename.endswith('.jpg') or filename.endswith('.png')):
        print(os.path.join(path,filename)) #
        img1=cv2.imread(os.path.join(path,filename), 0)
        #print('%d pixel' %(img1.shape[0]*img1.shape[1]) )
        if img1 is None:
            print('img1 is None')
            continue
        if img1.shape[0]*img1.shape[1]>50000000:
            print ('Too big picture > 50000000pixel |%s|%s' %(img1.shape,filename))
            continue
        kp1, des1 = sift.detectAndCompute(img1,None)        
        #print('%4d features|%s|%s' % (len(kp1),img1.shape,filename))
        cnt_time-=time.time()
        matches = bf.knnMatch(des0,des1, k=2)
        #print('%s' %(len(matches)) )
        '''
        for i in range(len(matches)):
            print (' %4d %4d %d %d %s %s' %(matches[i][0].queryIdx,matches[i][0].trainIdx,matches[i][0].imgIdx,matches[i][0].distance,kp0[matches[i][0].queryIdx].pt,kp1[matches[i][0].trainIdx].pt))
        '''
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append([m])
        #output success features
        #print len(good)
        print('%4d/%4d/%5d features|%s|%s' % (len(good),len(matches),len(kp1),img1.shape,filename) )
        
        #output success filename
        if len(good)>3:
            #print('%4d/%4d/%5d features|%s|%s' % (len(good),len(matches),len(kp1),img1.shape,filename) )
            print('%s' % (filename) )
        cnt_time+=time.time()
        '''    
        #output success features detail
        for i in range(len(good)):
            print (' %4d %4d %d %d %s %s' %(good[i][0].queryIdx,good[i][0].trainIdx,good[i][0].imgIdx,good[i][0].distance,kp0[good[i][0].queryIdx].pt,kp1[good[i][0].trainIdx].pt))
        '''

end_time=time.time()
print('%.3gs' %(end_time-start_time))
print('%.3gs' %(cnt_time))
#os.system("pause")