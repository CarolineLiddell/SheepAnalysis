

import numpy as np
import pandas as pd
import math
import LatLon
import glob
import os
import datetime
import matplotlib.pyplot as plt


### 1.DETERMINE START AND END TIME FOR ALL SHEEP

# import data
sheep = pd.read_csv("C:/Users/cl516/Dropbox/Sheep_Files_1/PyHM_Bainbridge1.csv", skipinitialspace=True)
#sheep = pd.read_csv("C:/Users/cl516/Dropbox/Sheep_Files_1/PyHM_Kate1.csv", skipinitialspace=True)
#sheep = pd.read_csv("C:/Users/cl516/Dropbox/Sheep_Files_1/PyHM_Amos1.csv", skipinitialspace=True)
#sheep = pd.read_csv("C:/Users/cl516/Dropbox/Sheep_Files_1/PyHM_Redlands1.csv", skipinitialspace=True)


# convert hours and minutes to strings of length = 2 characters
# need to be strings to create timestamp
sheep['Hours'] = sheep['Hours'].astype('S2')
sheep['Minutes'] = sheep['Minutes'].astype('S2')

# create a timestamp from the date, hours and minutes columns
sheep['timestamp'] = pd.to_datetime(sheep['Date'] + ' ' + sheep['Hours'] + ':' + sheep['Minutes'],
     format="%d/%m/%Y %H:%M")

# create empty lists to collect first and last GPS collar recordings of each sheep
start_last = []
end_first = []

# loop through all the sheep finding the maximum and minumum time stamp, add these to the lists
for sheepName in np.unique(sheep['Sheep_ID']):    
    min_t = (min(sheep['timestamp'][sheep['Sheep_ID']==sheepName]))     
    start_last.append(min_t)
    max_t = (max(sheep['timestamp'][sheep['Sheep_ID']==sheepName]))
    end_first.append(max_t)

# determine latest first GPS recording (start) and earliest last GPS recording (end)
start = max(start_last)
end = min(end_first)




### 2. CALCULATE DISTANCE TRAVELLED BY EACH SHEEP 

#filenames = glob.glob("C:/Phd/Alice/Btest/Bainbridge*.csv")
#allsheep = pd.read_csv("C:/Phd/Alice/Summary_Btest.csv", delimiter = ',')

#filenames = glob.glob("C:/Users/cl516/Dropbox/Sheep_Files_1/Amos1_edit/Amos*.csv")
#allsheep = pd.read_csv("C:/Users/cl516/Dropbox/Sheep_Files_1/Summary_Amos.csv", delimiter = ',')

#filenames = glob.glob("C:/Users/cl516/Dropbox/Sheep_Files_1/Bainbridge1/Bainbridge*.csv")
#allsheep = pd.read_csv("C:/Users/cl516/Dropbox/Sheep_Files_1/Summary_Bainbridge.csv", delimiter = ',', skipinitialspace=True)

filenames = glob.glob("C:/Users/cl516/Dropbox/Sheep_Files_1/Kate1/Kate*.csv")
allsheep = pd.read_csv("C:/Users/cl516/Dropbox/Sheep_Files_1/Summary_Kate.csv", delimiter = ',')

#filenames = glob.glob("C:/Users/cl516/Dropbox/Sheep_Files_1/Redlands1/Redlands*.csv")
#allsheep = pd.read_csv("C:/Users/cl516/Dropbox/Sheep_Files_1/Summary_Redlands.csv", delimiter = ',')

allsheep['Sheep_ID'] = allsheep['Sheep_ID '] 
data3 = pd.DataFrame()

# calculate distance and heading from GPS tracks  

for idx, f in enumerate(filenames):
    # remove spacing from headers
    data = pd.read_csv(f, skipinitialspace=True) 
    # extract the extension (.csv) from a filename, so that sheepname can equal filename    
    sheepname = os.path.basename(f)     # returns the base (end) of the path i.e. sheepname.csv
    sheepname = sheepname[0:-4]         # removes the last 4 characters (".csv") from name
    # sheepname = sheepname[0:-9]         # removes the last 9 characters ("_edit.csv") from name         
    FEC = allsheep.loc[allsheep['Sheep_ID']==sheepname].FEC.values[0]
    print("Processing %s" % sheepname)    
            
    # set start coordinates as those in row 0 of the first sheep (idx == 0; reference point for all sheep)
    if idx == 0:
        startlat = data.iloc[0].Latitude
        startlon = data.iloc[0].Longitude

    # create an empty array for distance and heading (no. rows, no. columns)
    # number of rows = len(data): as many rows as the dataframe "data" already has
    dis = np.empty((len(data), 1)) 
    head = np.empty((len(data), 1))
    start = LatLon.LatLon(startlat, startlon) 
    
    for index in xrange(len(data)):   
        # distance        
        walk = LatLon.LatLon(data.Latitude[index], data.Longitude[index])
        dis[index] = start.distance(walk)
        data['calc_distance'] = dis*1000
        
       # heading        
        heading = LatLon.LatLon(data.Latitude[index], data.Longitude[index])
        head[index] = math.radians(90-start.heading_initial(heading))
        data['calc_heading'] = head
    
   
    # strip minutes and hours from time and create new columns for them
    mins = np.array([datetime.datetime.strptime(a, '%H:%M:%S').minute for a in data['Time'].values])
    hours = np.array([datetime.datetime.strptime(a, '%H:%M:%S').hour for a in data['Time'].values])
    data['Minutes'] = mins
    data['Hours'] = hours
    
    # group data by averaging everything per minute    
    data2 = data.groupby(['Hours', 'Minutes'],as_index=False).mean()
    data2['Sheep_ID'] = sheepname
    data2['Date'] = data['Date']

    # group data per 5 minutes
    grouped = data2.groupby(data2.index / 5).mean()
    grouped['Sheep_ID'] = sheepname    
    
    # append grouped data of each sheep into the same file
    data3 = data3.append(grouped)    



### 3. DETERMINE DISTANCE MOVED IN 5 MINUTE BLOCKS

sheeps = np.unique(data3.loc[:, 'Sheep_ID'])

ID = []
resting = []
grazing = []
moving = []
FEC = []

for sheep in sheeps:
    rest = data3.loc[(data3.Sheep_ID == sheep) & (data3.calc_distance <= 5)]
    graze = data3.loc[(data3.Sheep_ID == sheep) & (data3.calc_distance > 5) & (data3.calc_distance <= 100)]
    move = data3.loc[(data3.Sheep_ID == sheep) & (data3.calc_distance > 100)]
    
    eggs = allsheep.loc[allsheep.Sheep_ID == sheep].FEC.values[0]
    
    r = len(rest)*5    
    g = len(graze)*5
    m = len(move)*5

    ID.append(sheep)
    resting.append(r)
    grazing.append(g)
    moving.append(m)
    FEC.append(eggs)

final = pd.DataFrame({'Sheep_ID' : ID, 'resting' : resting, 'grazing' : grazing, 'moving' : moving, 'FEC' : FEC})

# Plot the results
# Scatter plots           
plt.figure(1)
plt.subplot(131)
plt.scatter(final.FEC, final.grazing)
plt.xlabel("FEC")
plt.ylabel("Time spent Grazing")   

plt.subplot(132)
plt.scatter(final.FEC, final.moving)
plt.xlabel("FEC")
plt.ylabel("Time spent Moving") 

plt.subplot(133)
plt.scatter(final.FEC, final.resting)
plt.xlabel("FEC")
plt.ylabel("Time spent Resting") 


# Bar graphs
mean_grazing_highFEC = final.loc[final.FEC > 400].grazing.mean()
mean_moving_highFEC = final.loc[final.FEC > 400].moving.mean()
mean_resting_highFEC = final.loc[final.FEC > 400].resting.mean()

mean_grazing_mediumFEC = final.loc[(final.FEC <= 400) & (final.FEC > 100)].grazing.mean()
mean_moving_mediumFEC = final.loc[(final.FEC <= 400) & (final.FEC > 100)].moving.mean()
mean_resting_mediumFEC = final.loc[(final.FEC <= 400) & (final.FEC > 100)].resting.mean()

mean_grazing_lowFEC = final.loc[final.FEC <= 100].grazing.mean()
mean_moving_lowFEC = final.loc[final.FEC <= 100].moving.mean()
mean_resting_lowFEC = final.loc[final.FEC <= 100].resting.mean()

plt.figure(2)
n_groups = 3

means_highFEC = (mean_grazing_highFEC, mean_moving_highFEC, mean_resting_highFEC)
means_mediumFEC = (mean_grazing_mediumFEC, mean_moving_mediumFEC, mean_resting_mediumFEC)
means_lowFEC = (mean_grazing_lowFEC, mean_moving_lowFEC, mean_resting_lowFEC)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

rects1 = plt.bar(index, means_highFEC, bar_width,
                 color='b',
                 label='High FEC')

rects2 = plt.bar(index + bar_width, means_mediumFEC, bar_width,
                 color='y',
                 label='Medium FEC')

rects3 = plt.bar(index + 2*bar_width, means_lowFEC, bar_width,
                 color='g',
                 label='Low FEC')


plt.ylabel('Time')
plt.xticks(index + bar_width, ('Grazing', 'Moving', 'Resting'))
plt.legend()