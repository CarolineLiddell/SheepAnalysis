
import cv2
import numpy as np
import pandas as pd
import os
import re
import math
import time


# folder where movies are stored
DATADIR = '/home/ctorney/data/sheep/'

# file which has the list of movies to process
CLIPLIST = '/home/ctorney/workspace/SheepAnalysis/Tracking/clipList.csv'

# open up the list of movies
df = pd.read_csv(CLIPLIST)
df['clipname']=''

warp_mode = cv2.MOTION_EUCLIDEAN
number_of_iterations = 20

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = -1e-16;

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)


# loop through every movie in the list
for index, row in df.iterrows():
    

    # get movie start and stop time in seconds
    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    # set filenames for input and for saving the stabilized movie
    inputName = DATADIR + row.folder + '/' + row.filename
    outputName = time.strftime("%Y%m%d", time.strptime(row.date,"%d-%b-%Y")) + '-' + str(index) + '.avi'

    df.loc[index,'clipname'] = outputName
    
    
    print('Movie ' + row.folder + '/' + row.filename + ' from ' + str(timeStart) + ' to ' + str(timeStop) + ' out to ' + outputName)

    # open the video
    cap = cv2.VideoCapture(inputName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    # set the start and stop time in frames
    fStart = timeStart*fps
    fStop = timeStop*fps

    # go to the start of the clip
    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    # reduce to 120 frames a second - change number to required frame rate
    ds = math.ceil(fps/120)
    out = cv2.VideoWriter(DATADIR + row.folder + '/' + outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps/ds, S, True)

  
    im1_gray = np.array([])
    first = np.array([])

    warp_matrix = np.eye(2, 3, dtype=np.float32) 
    full_warp = np.eye(3, 3, dtype=np.float32)
    for tt in range(fStart,fStop):
        # Capture frame-by-frame
        _, frame = cap.read()
        if (tt%ds!=0):
            continue
        if not(im1_gray.size):
            # enhance contrast in the image
            im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            first = frame.copy()
        
        im2_gray =  cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        

        try:
            # find difference in movement between this frame and the last frame
            (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)    
        except cv2.error as e:
            im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            first = frame.copy()
        # this frame becames the last frame for the next iteration
        im1_gray =im2_gray.copy()
        
        # alll moves are accumalated into a matrix
        full_warp = np.dot(full_warp, np.vstack((warp_matrix,[0,0,1])))
        # create an empty image like the first frame
        im2_aligned = np.empty_like(frame)
        np.copyto(im2_aligned, first)
        # apply the transform so the image is aligned with the first frame and output to movie file
        im2_aligned = cv2.warpAffine(frame, full_warp[0:2,:], (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_LINEAR  , borderMode=cv2.BORDER_TRANSPARENT)
        out.write(im2_aligned)
        

    cap.release()
    out.release()
    
# save the cliplist with all the output file names
df.to_csv(CLIPLIST,index=False)


