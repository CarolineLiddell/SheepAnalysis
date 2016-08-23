# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:09:37 2016

@author: cl516
"""

import cv2
import numpy as np
import pandas as pd
import os
import re
import math
import time


# folder where movies and background images are stored
DATADIR = 'C:/Users/cl516/Dropbox/Sheep_Files_1/Videos/'

# file which has the list of movies to process
CLIPLIST = 'C:/PhD/Python/SheepAnalysis/Tracking/clipList.csv'

# open up the list of movies
df = pd.read_csv(CLIPLIST)

HD = os.getenv('HOME')

cv2.ocl.setUseOpenCL(False)

df = pd.read_csv(CLIPLIST)

for index, row in df.iterrows():
    


    # set filenames for input and for saving the output video
    inputName = DATADIR + row.folder + '/' + row.clipname
    outputName = time.strftime("%Y%m%d", time.strptime(row.date,"%d-%b-%Y")) + '-' + 'locateObjects' + '.avi'

    # set name for background image
    noext, ext = os.path.splitext(inputName)    
    background = noext + '.png'

    # open the video
    cap = cv2.VideoCapture(inputName)
    fCount = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    # open the background image
    img = cv2.imread(background)
    
    # compare video to background
    
    
    # save the video
    S = (1920,1080)
    fps = 6    
    out = cv2.VideoWriter(DATADIR + row.folder + '/' + outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps, S, True)
    