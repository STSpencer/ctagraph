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

K = keras.backend


def one_hot_to_labels(labels_one_hot):
    return np.sum([(labels_one_hot[:, i] == 1) * (i + 1) for i in range(4)], axis=0)


def get_edge_graph_in_model(input_data, model, sample_idx=0):
    layer_names = [l.name for l in model.layers if "edge_conv" in l.name]
    coord_mask = [np.sum(np.linalg.norm(inp_d[sample_idx], axis=-1)) == 500 for inp_d in input_data]
    assert True in coord_mask, "For plotting the spherical graph of the cosmic ray on at least one input has to have 3 dimensions XYZ"
    fig, axes = plt.subplots(ncols=len(layer_names), figsize=(5 * len(layer_names), 6))
    for i, lay_name in enumerate(layer_names):
        points_in, feats_in = model.get_input_at(0)
        coordinates = model.get_layer(lay_name).get_input_at(0)
        functor = K.function([points_in, feats_in], coordinates)
        sample_input = [inp[np.newaxis, sample_idx] for inp in input_data]
        try:
            layer_points, layer_features = functor(sample_input)
        except ValueError:
            layer_points = functor(sample_input)
        layer_points = np.squeeze(layer_points)
        adj = kneighbors_graph(layer_points, model.get_layer(lay_name).K)
        g = nx.from_scipy_sparse_matrix(adj)
        for c, s in zip(coord_mask, sample_input):
            if c == True:  # no 'is' here!
                pos = s
                break
        axes[i].set_title("XY-Projection of Graph formed in %s" % lay_name)

        nx.draw(
            g,
            cmap=plt.get_cmap('viridis'),
            pos=pos.squeeze()[:, :-1],
            node_size=10,
            width=0.3,
            ax=axes[i])
        axes[i].axis('equal')
    fig.tight_layout()
    return fig


def plot_eigenvectors(A):
    from scipy.sparse.linalg import eigsh # assumes L to be symmetricΛ,V = eigsh(L,k=20,which=’SM’) # eigen-decomposition (i.e. find Λ,V)
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
        axes[i].imshow(eigen.reshape(28,28))
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
    x = np.arange(len(loss))
    axes[0].plot(x, loss, c="navy")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Loss")
    axes[1].plot(x, acc, c="firebrick")
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
    X_train, y_train, X_val, y_val, X_test, y_test, A = mnist.load_data(k)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    n_x, n_y = (28, 28)
    x = np.linspace(0, 10, n_x)
    y = np.linspace(0, 10, n_y)
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
fig.savefig("./mnist_graph.png")

# plot first 20 eigenvectors of graph / most important eigenvectors in fourierdomain
fig = plot_eigenvectors(A)
fig.savefig("./mnist_eigen.png")

# Parameters
channels = 16           # Number of channels in the first layer
cheb_k = 2              # Max degree of the Chebyshev approximation
support = cheb_k + 1    # Total number of filters (k + 1)
learning_rate = 1e-2
epochs = 2
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
flatten = layers.Flatten()(graph_conv_2)
output = layers.Dense(10, activation='softmax')(flatten)

# Build model
model = models.Model(inputs=[X_in] + fltr_in, outputs=output)
optimizer = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()


A_train = [np.repeat(f[np.newaxis,...], batch_size, axis=0) for f in fltr]  # create batch of adjacency matrices as ht graph is constant for all samples!

print('Train model.\n') # Train model
history = []
for epoch in range(epochs):

    print("epoch", epoch, "\n")
    progbar = tensorflow.keras.utils.Progbar(N_samples, verbose=1)  # create fancy keras progressbar
    for i in range(N_samples // batch_size):

        beg = i*batch_size
        end = (i+1) * batch_size
        loss, acc = model.train_on_batch([X_train[beg:end]] + A_train, y_train[beg:end])
        progbar.add(batch_size, values=[("train loss", loss), ("acc", acc)])  # update fancy keras progress bar

        history.append([loss, acc])


fig = plot_history(history)
fig.savefig("./history_mnist.png")


print('Evaluating model.\n')  # Evaluate model

progbar = tensorflow.keras.utils.Progbar(test_samples, verbose=1)  # create fancy keras progressbar
for a in range(test_samples // batch_size):

    beg = a*batch_size
    end = (a+1) * batch_size
    eval_loss, eval_acc = model.test_on_batch([X_test[beg:end]] + A_train, y_test[beg:end])
    progbar.add(batch_size, values=[("eval loss", eval_loss), ("acc", eval_acc)])  # update fancy keras progress bar
