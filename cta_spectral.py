import scipy.sparse as sp

from sklearn.model_selection import train_test_split

from sklearn.neighbors import kneighbors_graph

#from tensorflow.keras.datasets import mnist as m

import h5py

import numpy as np

import glob

import sys


MNIST_SIZE = 48


#from tensorflow.keras.datasets import mnist as m

filelist = sorted(glob.glob('/home/ir-jaco1/rds/rds-iris-ip007/ir-jaco1/Data/*_0.hdf5')) #Only load in files with gammaflag=0                                                                                                                                                                                             


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


def load_data(k=8, noise_level=0.0):

    """                                                                                                                                                                                             

    Loads the MNIST dataset and a K-NN graph to perform graph signal                                                                                                                                

    classification, as described by [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).                                                                                                   

    The K-NN graph is statically determined from a regular grid of pixels using                                                                                                                     

    the 2d coordinates.                                                                                                                                                                             

                                                                                                                                                                                                    

    The node features of each graph are the MNIST digits vectorized and rescaled                                                                                                                    

    to [0, 1].                                                                                                                                                                                      

    Two nodes are connected if they are neighbours according to the K-NN graph.                                                                                                                     

    Labels are the MNIST class associated to each sample.                                                                                                                                           

                                                                                                                                                                                                    

    :param k: int, number of neighbours for each node;                                                                                                                                              

    :param noise_level: fraction of edges to flip (from 0 to 1 and vice versa);                                                                                                                     

                                                                                                                                                                                                    

    :return:                                                                                                                                                                                        

        - X_train, y_train: training node features and labels;                                                                                                                                      

        - X_val, y_val: validation node features and labels;                                                                                                                                        

        - X_test, y_test: test node features and labels;                                                                                                                                            

        - A: adjacency matrix of the grid;                                                                                                                                                          

    """

    A = _mnist_grid_graph(k)

    A = _flip_random_edges(A, noise_level).astype(np.float32)


#    (X_train, y_train), (X_test, y_test) = m.load_data()

#    X_train, X_test = X_train / 255.0, X_test / 255.0 This is a normalization term, comment it out for the time being but remember it's there.

    x,y=generate_training_data(filelist)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=10000)

    X_train = X_train.reshape(-1, MNIST_SIZE ** 2)

    X_test = X_test.reshape(-1, MNIST_SIZE ** 2)


    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=10000)


    return X_train, Y_train, X_val, Y_val, X_test, Y_test, A




def _mnist_grid_graph(k):

    """                                                                                                                                                                                             

    Get the adjacency matrix for the KNN graph.                                                                                                                                                     

    :param k: int, number of neighbours for each node;                                                                                                                                              

    :return:                                                                                                                                                                                        

    """

    X = _grid_coordinates(MNIST_SIZE)

    A = _get_adj_from_data(

        X, k, mode='connectivity', metric='euclidean', include_self=False

    )

    print("Returning A...")
    print(A)
    return A

