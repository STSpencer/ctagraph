import h5py
import numpy as np
import glob

filelist = sorted(glob.glob('/mnt/extraspace/exet4487/pointrun3/*.hdf5'))

filelist=filelist[:5]

def generate_training_data(filelist):
    """ Generates training/test sequences on demand
    """

    allcharges=[]
    alllabels=[]
    for file in filelist:
        inputdata = h5py.File(file, 'r')
        print(np.shape(inputdata['event_label']))
        for j in np.arange(np.shape(inputdata['event_label'])[0]):
            chargearr = inputdata['squared_training'][j, 0, :, :]
            labelsarr = inputdata['event_label'][j]
            allcharges.append(chargearr)
            alllabels.append(labelsarr)
    allcharges=np.asarray(allcharges)
    alllabels=np.asarray(alllabels)
    return allcharges, alllabels

x,y=generate_training_data(filelist)
print(np.shape(x),np.shape(y))
print(x[0])
print(y)
