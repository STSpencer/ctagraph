'''Script to import sim_telarray GCT-S simulations,
optionally plot them, and export them to hdf5 files. Designed to mix
proton, gamma datafiles together for CTANN training/testing.
Needs ctapipe, remember to use source activate cta-dev in advance.
This version includes randomization of the two datasets
 and the storing of parameterized waveforms.
Written by S.T. Spencer (samuel.spencer@physics.ox.ac.uk) 8/8/2018'''

import time
import matplotlib as mpl
import ctapipe
from ctapipe.io import event_source
from matplotlib.colors import LogNorm
from time import sleep
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.calib import CameraCalibrator
from ctapipe.core import Tool
from matplotlib import pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as anim
import time
from scipy.io import savemat
import sys
from astropy.io import fits
import h5py
from traitlets import (Integer, Float, List, Dict, Unicode)
from ctapipe.image import tailcuts_clean, dilate
from traitlets import Int
import scipy.signal as signals
from ctapipe.io import EventSeeker
from ctapipe.io import SimTelEventSource
from scipy.interpolate import UnivariateSpline
import os
import signal
from scipy.interpolate import splrep, sproot, splev
import astropy.units as unit
import numba
from numba import jit
from ctapipe.visualization import CameraDisplay 
from ctapipe.image.geometry_converter import chec_to_2d_array
from ctapipe.instrument import CameraGeometry

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class MultiplePeaks(Exception):
    pass

class NoPeaksFound(Exception):
    pass

def sig_handler(signum, frame):
    print("segfault")

signal.signal(signal.SIGSEGV, sig_handler)

# Run Options
# Import raw sim_telarray output files
runno = 1
gamma_data = "/store/adisims2/gamma/run" + str(runno) + ".simtel.gz"
hadron_data = "/store/adisims2/proton/run" + str(runno) + ".simtel.gz"

'''
from ctapipe.io import event_source
from ctapipe.calib import CameraCalibrator
import matplotlib.pyplot as plt
cal = CameraCalibrator()
source = event_source('/store/adisims/gamma/run50.simtel.gz')
event = next(iter(source)) 
cal(event)
geom = event.inst.subarray.tel[1].camera
disp = CameraDisplay(geom, image= event.dl1.tel[1].image) 
plt.show()
plt.savefig()

'''
gammaflag = int(input("Plot hadrons (0) or gammas (1)?: "))  # Should be 0 to plot hadrons or 1 for gammas.
plotev = False  # Whether or not to make animation plots for one single event.
event_plot = 0  # Min event number to plot
chan = 0  # PM Channel to use.
output_filename = '/store/adisims2/Data/hdftest'  # HDF5 files output name.


# Max number of events to read in for each of gammas/protons for training.
maxcount = 500
no_files = 10  # Number of files in which to store events
filerat = maxcount / no_files

print('Filerat', filerat)
no_tels = 1  # Number of telescopes

def pos_to_index(pos, size):
    rnd = np.round((pos / size).to_value(unit.dimensionless_unscaled), 1)
    unique = np.sort(np.unique(rnd))
    mask = np.append(np.diff(unique) > 0.5, True)
    bins = np.append(unique[mask] - 0.5, unique[-1] + 0.5)
    return np.digitize(rnd, bins) - 1

@jit
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



cmaps = [plt.cm.jet, plt.cm.winter,
         plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth, plt.cm.hot,
         plt.cm.cool, plt.cm.coolwarm]

hists = {}
chan = 0  # which channel to look at

count = 1  # Keeps track of number of events processed
key=0

caliber = CameraCalibrator()


# Read in gammas/ protons from simtel for each output file.
starttime=time.time()
for fileno in np.arange(1, no_files + 1):

    # Basic principle is to load in, calibrate and parameterize the gamma ray
    # events, then do the same for the protons. Then mix the two together and
    # write them to disk.

    # Initialize lists for hdf5 storage.
    to_matlab = {
        'id': [],
        'event_id': [],
        'label': [],
        'mc_energy':[],
        'tel_data': [],
        'tel_integrated': [],
        'raw_image': []}

    smallfile_name = output_filename + "run" + str(runno) + "_" + str(fileno) + "_gammaflag_" + str(gammaflag) + '.hdf5'

    gamma_source = SimTelEventSource(input_url=gamma_data,back_seekable=True)

    print('Processing Gammas')
    evfinder = EventSeeker(reader=gamma_source)
    # Determine events to load in using event seeker.
    startev = int(filerat * fileno - filerat)
    midev = int(filerat * fileno - filerat / 2.0)
    endev = int(filerat * fileno)
    print(startev, endev)

    for event in evfinder[startev:endev]:
        caliber(event)

        if count % 1000 == 0:
            print(count)

        if plotev == True and gammaflag == 0:
            break

        to_matlab['id'].append(count)
        to_matlab['event_id'].append(str(event.r0.event_id) + '01')
        to_matlab['label'].append(0.0)
        energy=event.mc.energy.to(unit.GeV)
        print(energy.value)
        to_matlab['mc_energy'].append(energy.value)

        # Initialize arrays for given event
        integrated = np.zeros((no_tels, 48, 48))
        raw = np.zeros((no_tels,2048))
        
        for index, tel_id in enumerate(event.dl0.tels_with_data):
            # Loop through all triggered telescopes.
            cam = event.inst.subarray.tel[tel_id].camera
            size = np.sqrt(cam.pix_area)
            col = pos_to_index(cam.pix_x, size)
            row = pos_to_index(cam.pix_y, size)
            integ_charges = event.dl1.tel[tel_id]['image']
            raw[tel_id-1,:]=integ_charges
            squared = np.full((row.max() + 1, col.max() + 1), np.nan)
            squared[row, col] = integ_charges
            integrated[tel_id - 1, :, :] = squared

            if plotev==True and gammaflag==1:
                plt.imshow(squared)
                plt.show()
                plt.savefig('gamma' + str(runno) + '.png')



        # Send to hdf5 writer lists.
        # List of triggered telescopes
        to_matlab['tel_integrated'].append(integrated)
        to_matlab['raw_image'].append(raw)
        count = count + 1


    # Read in protons from simtel
    print('Processing Protons')

    proton_hessfile = SimTelEventSource(input_url=hadron_data,back_seekable=True)
    evfinder = EventSeeker(reader=proton_hessfile)
    print(startev, endev)

    for event in evfinder[startev:endev]:
     
        if gammaflag == 1:
           break
        
        caliber = CameraCalibrator()
        caliber(event)
        if count % 1000 == 0:
            print(count)
        to_matlab['id'].append(int(count))
        to_matlab['event_id'].append(str(event.r0.event_id) + '02')
        to_matlab['label'].append(1.0)
        energy=event.mc.energy.to(unit.GeV)
        print(energy.value)
        to_matlab['mc_energy'].append(energy.value)

        # Create arrays for event.
        integrated = np.zeros((no_tels, 48, 48))
        raw = np.zeros((no_tels,2048))

       
        for index, tel_id in enumerate(event.dl0.tels_with_data):
            # Loop through all triggered telescopes.
            cam = event.inst.subarray.tel[tel_id].camera
            size = np.sqrt(cam.pix_area)
            col = pos_to_index(cam.pix_x, size)
            row = pos_to_index(cam.pix_y, size)
            integ_charges = event.dl1.tel[tel_id]['image']
            raw[tel_id-1,:]=integ_charges
            squared = np.full((row.max() + 1, col.max() + 1), np.nan)
            squared[row, col] = integ_charges
            integrated[tel_id - 1, :, :] = squared

            if plotev==True and gammaflag==0:
                plt.imshow(squared)
                plt.show()
                plt.savefig('proton' + str(runno) + '.png')

        # Send to hdf5 writer lists.
        # List of triggered telescopes
        to_matlab['tel_integrated'].append(integrated)
        to_matlab['raw_image'].append(raw)
        count = count + 1

    # Make everything arrays in order to randomize.
    to_matlab['id'] = np.asarray(to_matlab['id'])
    to_matlab['event_id'] = np.asarray(to_matlab['event_id'])
    to_matlab['label'] = np.asarray(to_matlab['label'])
    to_matlab['mc_energy'] = np.asarray(to_matlab['mc_energy'])
    to_matlab['tel_integrated'] = np.asarray(to_matlab['tel_integrated'])
    to_matlab['raw_image'] = np.asarray(to_matlab['raw_image'])

    no_events = len(to_matlab['label'])
    randomize = np.arange(len(to_matlab['label']), dtype='int')

    # Implement uniform randomization here
    np.random.shuffle(randomize)
    to_matlab['id'] = to_matlab['id'][randomize]
    to_matlab['event_id'] = to_matlab['event_id'][randomize]
    to_matlab['label'] = to_matlab['label'][randomize]
    to_matlab['mc_energy'] = to_matlab['mc_energy'][randomize]
    to_matlab['tel_integrated'] = to_matlab['tel_integrated'][randomize]
    to_matlab['raw_image'] = to_matlab['raw_image'][randomize]

    h5file = h5py.File(smallfile_name, "w")

    print('Writing')

    # HDF5 writer code
    lab_event = h5file.create_dataset(
        'event_label', (no_events,), dtype='i', compression="gzip")
    id_group = h5file.create_dataset(
        'id', (no_events,), dtype='i', compression="gzip")
    id_group2 = h5file.create_dataset(
        'event_id', (no_events,), dtype='i', compression="gzip")
    energy_group = h5file.create_dataset(
        'mc_energy', (no_events,), dtype='f', compression="gzip")

    squared_group = h5file.create_dataset(
        "squared_training", (no_events, 4, 48, 48), dtype='f', compression="gzip")
    raw_group = h5file.create_dataset(
        "raw_images", (no_events, 4, 2048), dtype='f', compression="gzip")

    for index in np.arange(0, no_events):
        index = int(index)
        lab_event[index] = np.int64(to_matlab['label'][index])
        id_group[index] = np.int64(to_matlab['id'][index])
        id_group2[index] = np.int64(to_matlab['event_id'][index])
        energy_group[index] = np.float64(to_matlab['mc_energy'][index])
        squared_group[index, :, :, :] = to_matlab['tel_integrated'][index]
        raw_group[index, :, :] = to_matlab['raw_image'][index]

    h5file.close()
endtime=time.time()
runtime=endtime-starttime
print('Time for 10 events to be written', runtime)
plt.show()
plt.savefig('finalfig' + str(runno) + '.png')
