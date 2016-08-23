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
    
# get movie start and stop time in seconds
    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    # set filenames for input and for saving the output video
    inputName = DATADIR + row.folder + '/' + row.clipname
    outputName = time.strftime("%Y%m%d", time.strptime(row.date,"%d-%b-%Y")) + '-' + 'locateObjects' + '.avi'

    df.loc[index,'clipname'] = outputName
    
    
    print('Movie ' + row.folder + '/' + row.filename + ' from ' + str(timeStart) + ' to ' + str(timeStop) + ' out to ' + outputName)

   # set name for background image
    noext, ext = os.path.splitext(inputName)    
    background = noext + '.png'

    # open the background image
    img = cv2.imread(background)


   # open the video
    cap = cv2.VideoCapture(inputName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    # set the start and stop time in frames
    fStart = int(timeStart*fps)
    fStop = int(timeStop*fps)

    # go to the start of the clip
    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    # reduce to 120 frames a second - change number to required frame rate
    ds = math.ceil(fps/6.0)
    out = cv2.VideoWriter(DATADIR + row.folder + '/' + outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps/ds, S, True)

  
    first = np.array([])

    warp_matrix = np.eye(2, 3, dtype=np.float32) 
    full_warp = np.eye(3, 3, dtype=np.float32)

    for tt in range(fStart,fStop):

        # Capture frame-by-frame
        _, frame = cap.read()    
        first = frame.copy()
        
        # create an empty image like the first frame
        im2 = np.empty_like(frame)
        np.copyto(im2, first)
        
        # output to movie file
        out.write(im2)
        
        
    cap.release()
    out.release()
    
    

    
    
    
    
   