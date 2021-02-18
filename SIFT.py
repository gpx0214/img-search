#!/usr/bin/env Python
# -*- coding:UTF-8 -*-
import numpy as np
import cv2
import os
import sys,getopt
from matplotlib import pyplot as plt
import time
import math

start_time=time.time()
try:
    fn0= sys.argv[1]
    fn1= sys.argv[2]
except:
    fn0 = 'D:\\20170105\\20161228184344.[1]0007.jpg'  #'D:\\IMG_20160124_144219_640_480.jpg' #'D:\\bluestacks_screen_shot170107-1.429coins50.jpg'
    fn1 = 'D:\\20170105\\20161228184344.[1]0008.jpg' #'D:\\IMG_20160124_144227_640_480.jpg' #'D:\\bluestacks_screen_shot170107-1.png'

print fn0
print fn1

sift = cv2.SIFT()

img0=cv2.imread(fn0, 0)
img1=cv2.imread(fn1, 0)

if img0 is not None:
    print img0.shape

if img1 is not None:
    print img1.shape

kp0, des0 = sift.detectAndCompute(img0,None)
kp1, des1 = sift.detectAndCompute(img1,None)
print('img1 - %d features, img2 - %d features' % (len(kp0), len(kp1)))
for i in range(len(kp0)):
    print('%s %s %s %s %s' %(kp0[i].pt,kp0[i].size,kp0[i].angle,kp0[i].response,kp0[i].octave))
print ' '
#for i in range(len(kp1)):
#    print('%s %s %s %s %s' %(kp1[i].pt,kp1[i].size,kp1[i].angle,kp1[i].response,kp1[i].octave))

bf = cv2.BFMatcher()
matches = bf.knnMatch(des0,des1, k=2)
print ('len(matches)=%s'%(len(matches)))
print ('len(matches[0])=%s'%(len(matches[0])))
for i in range(len(matches)):
    #print ('%4d %4d %d %d ' %(matches[i][0].queryIdx,matches[i][0].trainIdx,matches[i][0].imgIdx,matches[i][0].distance))
    print ('%4d %4d %d %d %s %s' %(matches[i][0].queryIdx,matches[i][0].trainIdx,matches[i][0].imgIdx,matches[i][0].distance,kp0[matches[i][0].queryIdx].pt,kp1[matches[i][0].trainIdx].pt))
#for i in range(len(matches)):
#    print (' %s %s %d %d ' %(kp0[matches[i][0].queryIdx].pt,kp1[matches[i][0].trainIdx].pt,matches[i][0].imgIdx,matches[i][0].distance))

good = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append([m])

print ('len(good)=%s' %(len(good)))
print ('len(good[0])=%s' %(len(good[0]))) #1
for i in range(len(good)):
    #print ('%4d %4d %d %d ' %(good[i][0].queryIdx,good[i][0].trainIdx,good[i][0].imgIdx,good[i][0].distance))
    print ('%4d %4d %d %d %s %s' %(good[i][0].queryIdx,good[i][0].trainIdx,good[i][0].imgIdx,good[i][0].distance,kp0[good[i][0].queryIdx].pt,kp1[good[i][0].trainIdx].pt))

def sift2svg(kp0,filename):
    print(filename)
    fsvg=open(filename,"w") #,encoding='utf-8'
    fsvg.write('<?xml version="1.0" standalone="no"?> \n')
    fsvg.write('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" \n')
    fsvg.write('"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n')
    fsvg.write('<svg width="1280" height="720" version="1.1"\n')
    fsvg.write('xmlns="http://www.w3.org/2000/svg">\n')
    for i in range(len(kp0)):
        x=kp0[i].pt[0]
        y=kp0[i].pt[1]
        r=kp0[i].size
        a=kp0[i].angle
        fsvg.write('<circle cx="%s" cy="%s" r="%s" stroke="gray" stroke-width="1" fill="None" />\n' %(x,y,r))
        fsvg.write('<line x1="%s" y1="%s" x2="%s" y2="%s" style="stroke:rgb(0,255,0);stroke-width:1" />\n' %(x,y,x+r*math.cos(math.radians(a)),y+r*math.sin(math.radians(a))))
    fsvg.write('</svg>\n')
    fsvg.close()


sift2svg(kp0,fn0+'.svg')
sift2svg(kp1,fn1+'.svg')


#img3 = cv2.drawMatchesKnn(img0,kp0,img1,kp1,good,flags=2)#opencv2.4.13
end_time=time.time()
print('%.3gs' %(end_time-start_time))
#os.system("pause")
