'''Script to count events in a file
Requires ctapipe to be activated
Adi Jacobson 16/11/2020'''

import ctapipe
from ctapipe.io import EventSeeker
from ctapipe.io import SimTelEventSource

#Open file
gamma_data = "/store/adisims/gamma/run1.simtel.gz"
file = SimTelEventSource(input_url=gamma_data,back_seekable=True)

#Loop through events in the file
#count = 0
evfinder = EventSeeker(reader = file)
print type(evfinder)
#for event in evfinder:
#    count += 1

#Print number of events
#print("There are %d events in run%d", count, runno)
