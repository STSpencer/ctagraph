'''Script to count events in a file
Requires ctapipe to be activated
Adi Jacobson 16/11/2020'''

import ctapipe
from ctapipe.io import EventSeeker
from ctapipe.io import SimTelEventSource

#Gammas
print("Gamma files:")

#Open file

for runno in range(1,2):

  gamma_data = "/store/adisims2/gamma/run" + str(runno) + ".simtel.gz"
  file = SimTelEventSource(input_url=gamma_data,back_seekable=True)

  evfinder = EventSeeker(reader = file)
  count = 0
#  count = len(evfinder)
  for event in evfinder:
    count += 1

#Print number of events
  print("There are %d events in run%d" % (count, runno))

#Protons
print("Proton files:")

#Open file

for runno in range(1,2):

  proton_data = "/store/adisims2/proton/run" + str(runno) + ".simtel.gz"
  file = SimTelEventSource(input_url=proton_data,back_seekable=True)

  evfinder = EventSeeker(reader = file)
  count = 0
#  count = len(evfinder)
  for even in evfinder:
    count += 1

  print("There are %d events in run%d" % (count, runno))
