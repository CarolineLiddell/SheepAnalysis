

import cv2
import numpy as np
import pandas as pd
import os
import re
import math
import time

# folder where movies are stored
DATADIR = '/home/ctorney/Dropbox/Sheep_Files_1/Videos/'
DATADIR = '/home/ctorney/data/sheep/'

# file which has the list of movies to process
CLIPLIST = '/home/ctorney/workspace/SheepAnalysis/Tracking/clipList.csv'

# open up the list of movies
df = pd.read_csv(CLIPLIST)

HD = os.getenv('HOME')

cv2.ocl.setUseOpenCL(False)

df = pd.read_csv(CLIPLIST)

for index, row in df.iterrows():
    


    # set filenames for input and for saving the background image
    inputName = DATADIR + row.folder + '/' + row.clipname
    noext, ext = os.path.splitext(inputName)
    outputName = DATADIR + row.folder + '/' + noext + '.png'

    
    

    # open the video
    cap = cv2.VideoCapture(inputName)
    fCount = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)


    # create the background builder
    pMOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=1,history=400)


    # build the background image
    for tt in range(fCount):
        _, frame = cap.read()
        fgmask = pMOG2.apply(frame)

    # save the image
    cv2.imwrite(outputName,pMOG2.getBackgroundImage() )
#    bkg = cv2.convertScaleAbs(pMOG2.getBackgroundImage() )
    

