import h5py
import numpy as np
import glob
import sys

filelist = sorted(glob.glob('/home/ir-spen1/rds/rds-iris-ip007/ir-jaco1/Data/*_0.hdf5')) #Only load in files with gammaflag=0                                                                                                                                                                                             


filelist=filelist[:10] #Restrict to only loading in 1 hdf5 file, change this cut to load in more                                                                                                                                                                                                                           


np.set_printoptions(threshold=sys.maxsize) #So that arrays actually get printed                                                                                                                                                                                                                                           


def cam_squaremaker(data):

    '''Function to translate CHEC-S integrated images into square arrays for                                                                                                                                                                                                                                              

    analysis purposes.'''

    square = np.zeros(2304)

    i = 0

    while i < 48:

        if i < 8:

            xmin = 48 * i + 8

            xmax = 48 * (i + 1) - 8

            square[xmin:xmax] = data[i * 32:(i + 1) * 32]

            i = i + 1

        elif i > 7 and i < 40:

            square[384:1920] = data[256:1792]

            i = 40

        else:

            xmin = 48 * i + 8

            xmax = 48 * (i + 1) - 8

            square[xmin:xmax] = data[512 + i * 32:544 + i * 32]

            i = i + 1


    square.resize((48, 48))

    square = np.flip(square, 0)

    return square


def generate_training_data(filelist):

    """ Generates training/test sequences on demand                                                                                                                                                                                                                                                                       

    """


    allcharges=[]

    alllabels=[]

    for file in filelist:

        inputdata = h5py.File(file, 'r')

        #print(np.shape(inputdata['event_label']))

        for j in np.arange(np.shape(inputdata['event_label'])[0]):

            chargearr = inputdata['raw_images'][j, 0, :]

            #print(chargearr, cam_squaremaker(chargearr),np.shape(chargearr)) #Uncomment to actually print out array values                                                                                                                                                                                               

            labelsarr = inputdata['event_label'][j]

            allcharges.append(cam_squaremaker(chargearr))

            alllabels.append(labelsarr)

    allcharges=np.asarray(allcharges)

    alllabels=np.asarray(alllabels)

    return allcharges, alllabels


x,y=generate_training_data(filelist)

#print(np.shape(x),np.shape(y))

#print(x[0])

#print(y)

print("Creating training and testing arrays")

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=10000)

print(X_train[0])
print(X_test[0])
print(Y_train[0])
print(X_test[0])


