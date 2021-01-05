import h5py
import numpy as np
import glob

filelist = sorted(glob.glob('/path/to/data/*.hdf5'))

def generate_training_data(filelist):
    """ Generates training/test sequences on demand
    """

    nofiles = 0
    i = 0  # No. events loaded in total

    for file in filelist:
        inputdata = h5py.File(file, 'r')
        chargearr = np.asarray(inputdata['squared_training'][:, 0, :, :])
        labelsarr = np.asarray(inputdata['event_label'][:])
        valilocs = np.where(labelsarr!=-1)[0]
        labelsarr = labelsarr[valilocs]
        chargearr = chargearr[valilocs,:,:]
        idarr = np.asarray(inputdata['id'][:])
        nofiles = nofiles + 1
        inputdata.close()
        chargearr = np.reshape(chargearr, (np.shape(chargearr)[0], 48, 48, 1)) 
        training_sample_count = len(chargearr)
        i = i + len(labelsarr)
        countarr = np.arange(0, len(labelsarr))
        ta2 = np.zeros((training_sample_count, 48, 48, 1))   
        ta2[:, :, :, 0] = chargearr[:, :, :, 0]
        trainarr = ta2
    return trainarr, labelsarr
