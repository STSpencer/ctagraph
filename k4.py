'''MNIST Chebyshev network example taken from vispa.rwth.aachen.de, original credit to Erdmann et. al group.'''
import tensorflow
from tensorflow.keras import models, layers, optimizers
from matplotlib import pyplot as plt
import networkx as nx
from spektral.layers import ChebConv
from spektral.utils.convolution import chebyshev_filter
import numpy as np
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from tensorflow import keras
import scipy.sparse as sp
from spektral.datasets import mnist
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import glob
import sys
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

runname=str(sys.argv[1])

K = keras.backend

MNIST_SIZE = 48

filelist = sorted(glob.glob('/home/ir-jaco1/rds/rds-iris-ip007/ir-jaco1/Data/*_0.hdf5')) #Only load in files with gammaflag=0                                                                                                                                                                                             
filelist=filelist[:50] #Restrict to only loading in 1 hdf5 file, change this cut to load in more                                                                                                                                                                                                                           
np.set_printoptions(threshold=sys.maxsize) #So that arrays actually get printed

def get_confusion_matrix_one_hot_tc(runname,model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,                                                                                                                                                                                 
    where truth is 0/1, and max along each row of model_results is model result                                                                                                                                                                                      
    '''
    mr=[]
    mr2=[]
    mr3=[]
    for x in model_results:
        mr.append(np.argmax(x))
        mr2.append(x)
        if np.argmax(x)==0:
            mr3.append(1-x[np.argmax(x)])
#            print("argmax(x) is 0 and 1-x[argmax(x)] is %d" % (1-x[np.argmax(x)]))
#            mr3.append(1 - np.argmax(x))
        elif np.argmax(x)==1:
            mr3.append(x[np.argmax(x)])
#            print("argmax(x) is 1 and x[np.argmax(x)] is %d" % x[np.argmax(x)])
#            mr3.append(np.argmax(x))
    model_results=np.asarray(mr)
    print("First 10 model_results")
    print(model_results[:10])
    truth=np.asarray(truth)
#    print("model_results")
#    print(len(np.rint(np.squeeze(model_results))))
#    print("truth")
#    print(len(truth))
#    print("y_predicted")
#    print(np.squeeze(model_results))
    print(np.rint(np.squeeze(model_results[:10])))
    
    cm=confusion_matrix(y_target=truth,y_predicted=np.rint(np.squeeze(model_results)),binary=True)
    fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(5,5))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('./Figures/'+runname+'confmat.png')
    fpr,tpr,thresholds=roc_curve(truth,np.asarray(mr3))
    plt.figure()
    lw = 2
    aucval=auc(fpr,tpr)
    print("aucval")
    print(aucval)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % aucval)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('./Figures/'+runname+'_roc.png')
    np.save('./confmatdata/'+runname+'_fp.npy',fpr)
    np.save('./confmatdata/'+runname+'_tp.npy',tpr)
    return cm

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

def _flip_random_edges(A, p_swap):
    if not A.shape[0] == A.shape[1]:
        raise ValueError("A must be a square matrix.")
    dtype = A.dtype
    A = sp.lil_matrix(A).astype(np.bool)
    n_elem = A.shape[0] ** 2
    n_elem_to_flip = round(p_swap * n_elem)
    unique_idx = np.random.choice(n_elem, replace=False, size=n_elem_to_flip)
    row_idx = unique_idx // A.shape[0]
    col_idx = unique_idx % A.shape[0]
    idxs = np.stack((row_idx, col_idx)).T
    for i in idxs:
        i = tuple(i)
        A[i] = np.logical_not(A[i])
    A = A.tocsr().astype(dtype)
    A.eliminate_zeros()
    return A

def generate_training_data(filelist):
    """ Generates training/test sequences on demand
    """
    allcharges=[]
    alllabels=[]
    events = 0
    for file in filelist:
        try:
            inputdata = h5py.File(file, 'r')
        except OSError:
            continue
        events += len(inputdata['event_label'])
        print("SHAPE IS: %d" % np.shape(inputdata['event_label']))
        for j in np.arange(np.shape(inputdata['event_label'])[0]):
            chargearr = inputdata['squared_training'][j, 0, : , :]
            #print(chargearr, cam_squaremaker(chargearr),np.shape(chargearr)) #Uncomment to actually print out array values                                                                                                                                                                                               
            labelsarr = inputdata['event_label'][j]
            chargearr=np.nan_to_num(chargearr)
#            chargearr.resize((48,48))
            allcharges.append(chargearr)
            alllabels.append(labelsarr)
    allcharges=np.asarray(allcharges)
    print("TOTAL NUMBER OF EVENTS IS: %d" % events)
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
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

#    X_train = (X_train-np.amin(X_train,axis=0))/(np.amax(X_train,axis=0)-np.amin(X_train,axis=0))
#    X_test = (X_test-np.amin(X_test,axis=0))/(np.amax(X_test,axis=0)-np.amin(X_test,axis=0))

    X_train = X_train.reshape(-1, MNIST_SIZE ** 2)
    X_test = X_test.reshape(-1, MNIST_SIZE ** 2)
#    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)
    y_val = y_test
    X_val = X_test
    print("y_val")
    print(len(y_val))
    print("X_test")
    print(len(X_test))
    return X_train, y_train, X_val, y_val, X_test, y_test, A

def _grid_coordinates(side):
    M = side ** 2
    x = np.linspace(0, 1, side, dtype=np.float32)
    y = np.linspace(0, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), np.float32)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z

def _get_adj_from_data(X, k, **kwargs):
    A = kneighbors_graph(X, k, **kwargs).toarray()
    A = sp.csr_matrix(np.maximum(A, A.T))

    return A

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


def one_hot_to_labels(labels_one_hot):
    return np.sum([(labels_one_hot[:, i] == 1) * (i + 1) for i in range(4)], axis=0)

def plot_eigenvectors(A):
    from scipy.sparse.linalg import eigsh # assumes L to be symmetricΛ,V = eigsh(L,k=20,which=SM) # eigen-decomposition (i.e. find Λ,V)
    # adapted from https://towardsdatascience.com/spectral-graph-convolution-explained-and-implemented-step-by-step-2e495b57f801
    N = A.shape[0] # number of nodes in a graph
    D = np.sum(A, 0) # node degrees
    D_hat = np.diag((D + 1e-5)**(-0.5)) # normalized node degrees
    L = np.identity(N) - np.dot(D_hat, A).dot(D_hat) # Laplacian
    _,V = eigsh(L,k=20,which='SM') # eigen-decomposition (i.e. find lambda,eigenvectors)
    Vs = [np.split(V, 20, axis=-1)][0]
    nrows = 4
    fig, axes = plt.subplots(nrows= nrows, ncols=int(np.ceil(len(Vs)/nrows)), figsize=(16,9))
    axes = axes.flatten()
    for i, eigen in enumerate(Vs):
        axes[i].imshow(eigen.reshape(48,48))
        axes[i].axis('equal')

    fig.tight_layout()
    return fig

def plot_history(history):
    fig, axes = plt.subplots(2, figsize=(12,8))
    if type(history) == dict:
        loss = history["loss"]
        acc = history["acc"]
    else:
        loss, acc = np.split(np.array(history), 2, axis=-1)

#    total_loss = 0
#    total_acc = 0
    av_loss = []
    av_acc = []
    bin_size = 1000
    total_loss = 0
    total_acc = 0

    for i in range(len(loss)):
        if i < bin_size:
            av_loss.append(loss[i])
        elif i > (len(loss)-bin_size):
            av_loss.append(loss[i])
        else:
            total_loss = 0
            for j in range(bin_size):
                total_loss += loss[i-j]
                total_loss += loss[i+j]
            total_loss -+ loss[i]
            av = total_loss / (2 * bin_size - 1)
            av_loss.append(av)


    for i in range(len(acc)):
        if i < bin_size:
            av_loss.append(acc[i])
        elif i > (len(acc)-bin_size):
            av_loss.append(acc[i])
        else:
            total_acc = 0
            for j in range(bin_size):
                total_acc += acc[i-j]
                total_acc += acc[i+j]
            total_acc -+ acc[i]
            av = total_acc / (2 * bin_size - 1)
            av_acc.append(av)

#    for item in loss:
#        total_loss += item
#        av_loss.append((total_loss/(len(av_loss)+1)))

#    for item in acc:
#        total_acc += item
#        av_acc.append((total_acc/(len(av_acc)+1)))
    
    x = np.arange(len(loss))
    y = np.arange(len(av_loss))
    z = np.arange(len(av_acc))
    axes[0].plot(x, loss, c="navy")
    axes[0].plot(y, av_loss, c="firebrick")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Loss")
    axes[1].plot(x, acc, c="firebrick")
    axes[1].plot(z, av_acc, c="navy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    if type(history) == dict:
        axes[0].set_xlabel("Epochs")
        axes[1].set_xlabel("Epochs")
    else:
        axes[0].set_xlabel("Iterations")
        axes[1].set_xlabel("Iterations")
    fig.tight_layout()
    return fig

def mnist_regular_graph(k=8):
    ''' load mnist dataset as graph. The graph is created using knn.
        You can change the number of neighbors per node by changing k.
        params:
            k: number of neighbors per node
        returns:
            X_train, y_train, X_val, y_val, X_test, y_test, A, node_positions
    '''
    X_train, y_train, X_val, y_val, X_test, y_test, A = load_data(k)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    n_x, n_y = (48, 48)
    x = np.linspace(0, 2, n_x)
    y = np.linspace(0, 2, n_y)
    y = np.flip(y)
    xv, yv = np.meshgrid(x, y)
    pos = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    return X_train, y_train, X_val, y_val, X_test, y_test, A.toarray(), pos

# Load data
X_train, y_train, X_val, y_val, X_test, y_test, A, pos = mnist_regular_graph(k=8)  # k=number of next neighbors

# plot example of single graph sample
fig, ax = plt.subplots(1, figsize=(9,9))
ax.axis('equal')

g = nx.from_numpy_matrix(A)
nx.draw(
    g,
    cmap=plt.get_cmap('jet'),
    pos=pos,
    node_size=100,
    node_color=X_train[5].squeeze(),  # set node color ~ to pixel intensity
    width=2,
    )

fig.tight_layout()
fig.savefig("./mnist_graph_"+runname+".png")

# plot first 20 eigenvectors of graph / most important eigenvectors in fourierdomain
fig = plot_eigenvectors(A)
fig.savefig("./mnist_eigen_"+runname+".png")

# Parameters
channels = 16           # Number of channels in the first layer
cheb_k = 3              # Max degree of the Chebyshev approximation
support = cheb_k + 1    # Total number of filters (k + 1)
learning_rate = 1e-2
epochs = 15
batch_size = 64

N_samples = X_train.shape[0]  # Number of samples in the training dataset
test_samples = X_test.shape[0]
N_nodes = X_train.shape[-2]   # Number of nodes in the graph


# Preprocessing operations
fltr = chebyshev_filter(A, cheb_k) # normalaize adjacency matrix


# Model definition
X_in = layers.Input(shape=(N_nodes, 1))

# One input filter for each degree of the Chebyshev approximation
fltr_in = [layers.Input((N_nodes, N_nodes)) for _ in range(support)]

# Architecture Definition

graph_conv_1 = ChebConv(channels,
                        activation='relu',
                        use_bias=False)([X_in] + fltr_in)
graph_conv_2 = ChebConv(2 * channels,
                        activation='softmax',
                        use_bias=False)([graph_conv_1] + fltr_in)
graph_conv_3 = ChebConv(2 * channels,
                        activation='softmax',
                        use_bias=False)([graph_conv_2] + fltr_in)
flatten = layers.Flatten()(graph_conv_3)
output = layers.Dense(2, activation='sigmoid')(flatten)

# Build model
model = models.Model(inputs=[X_in] + fltr_in, outputs=output)
optimizer = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              weighted_metrics=['acc'])
model.summary()


A_train = [np.repeat(f[np.newaxis,...], batch_size, axis=0) for f in fltr]  # create batch of adjacency matrices as ht graph is constant for all samples!

print('Train model.\n') # Train model
history = []
#TRYING HISTORY AS A DICTIONARY INSTEAD OF AN ARRAY
#history = {}
for epoch in range(epochs):

    print(epochs)

    print("epoch", epoch, "\n")
    progbar = tensorflow.keras.utils.Progbar(N_samples, verbose=2)  # create fancy keras progressbar
    for i in range(N_samples // batch_size):

        beg = i*batch_size
        end = (i+1) * batch_size
        loss, acc = model.train_on_batch([X_train[beg:end]] + A_train, y_train[beg:end])
        progbar.add(batch_size, values=[("train loss", loss), ("acc", acc)])  # update fancy keras progress bar

        history.append([loss, acc])


fig = plot_history(history)
fig.savefig("./history_mnist_"+runname+".png")
np.save("./"+runname+"history.npy", history)


print('Evaluating model.\n')  # Evaluate model

predictions=[]
progbar = tensorflow.keras.utils.Progbar(test_samples, verbose=2)  # create fancy keras progressbar
for a in range(test_samples // batch_size):
    
    beg = a*batch_size
    end = (a+1) * batch_size

    pred=model.predict_on_batch([X_test[beg:end]] + A_train)
    predictions.append(pred)
    print("New iteration")
    print("a")
    print(a)
    print("pred")
#    print(pred)
    print(pred.shape)
    print("predictions")
    print(len(predictions))

    
    eval_loss, eval_acc = model.test_on_batch([X_test[beg:end]] + A_train, y_test[beg:end])
    progbar.add(batch_size, values=[("eval loss", eval_loss), ("acc", eval_acc)])  # update fancy keras progress bar

    print("eval loss: %d, eval acc: %d" % (eval_loss, eval_acc))


predictions = np.asarray(predictions)
flat_predictions = list(np.vstack(predictions))
print("flat_predictions")
print(flat_predictions[:10])
#print(len(flat_predictions))
flat_predictions = flat_predictions[:len(y_val)]
y_val = y_val[:len(flat_predictions)]
#print(len(flat_predictions))
#print(flat_predictions)

get_confusion_matrix_one_hot_tc(runname, flat_predictions, y_val)



