# Useful function to reconsider:
#   .flatten()
#   .multiply() : elementwise multiplication
#   .eye() : genrate identity matrix
#   .dot() : matrix multiplication
#

import numpy as np
from scipy import optimize
import scipy as sci
import sys
from matplotlib import pyplot


def display_digit(X, N):
    """To display some digit from the data set

    Parameters
    ----------
    X : numpy.ndarray
        training data sets
    N : int
        pixel dimension of an image
    """
    for i in range(0,X.shape[0]):   
        image = X[i]
        image = np.array(image, dtype='float')   
        if N == 20:
            pixels = image.reshape((N,N), order='F')  
        else:
            pixels = image.reshape((N,N)) 
        pyplot.imshow(pixels, cmap='binary')   
        pyplot.show()
    pass

def rand_weight_init(layer_in, layer_out):
    """To break the symmetry on matrix of weight

    Parameters
    ----------
    layer_in : int,
        number of layers in the first layer
    layer_out : int
        number of layers in the second layer

    Returns
    -------
    W 
        matrix weight of size S_(j+1) X S_j + 1, where S is some layer of the neural networks
        such that every element in matrix is randomized such a way to provide a symmetry breaking
    """
    
    W = np.zeros((layer_out, layer_in + 1))
    # W = np.asmatrix(W)
    epsilon_init = 0.12
    W = np.random.rand(layer_out, 1 + layer_in) * \
        (2 * epsilon_init) - epsilon_init

    return W


def sigmoid(z):
    """To compute a logistic activation value for a node
    sigmoid = logitic activation function in neural networks
    
    Parameters
    ----------
    z : numpy.ndarray
        z(j) = theta(j-1) X a(j-1), such that:
        z is a vector 
        a is the value of "activation" of a all units (neurons) in layer j-1
        theta is the matrix of weight which control the function mapping from layer j-1 to layer j

    Returns
    -------
    g : numpy.ndarray
        activaltion value for each node in some layer j in a neural networks, represented as a matrix
    """
    # g = 1.0 / (1.0 + sci.special.expit(-z))
    # print(z)
    g = 1.0 / (1.0 + np.exp(-z))

    # if g.shape != (26,1):
    #     print(g.shape)
    return g


def sigmoid_gradient(z):
    """To compute a logistic activation value for a node
    sigmoid = logitic activation function in neural networks
    
    Parameters
    ----------
    z : numpy.ndarray
        z(j) = theta(j-1) X a(j-1), such that:
        z is a vector 
        a is the value of "activation" of a all units (neurons) in layer j-1
        theta is the matrix of weight which control the function mapping from layer j-1 to layer j

    Returns
    -------
    g : numpy.ndarray
        activaltion value for each node in some layer j in a neural networks, represented as a matrix
    """
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


# TODO: change theta_1 and theta_2 to an unrolled parameter
#       and unrolled them inside the function
def cost_function(
    weight_params,
    input_layer_size, hidden_layer_size, num_labels,
    X, y, lambda_val
):
    """To compute the cost function and the gradient/partial derivative
    
    Parameters
    ----------
    weight_params : numpy.ndarray
        unrolled matrix of matrices of weight
    
    input_layer_size : int
        the size of the input layer / the size of input units

    hidden_layer_size : int
        the size of the hidden layer / the number of hidden units

    num_labels : int
        the number of labels this represent digit 0 to 9 / the number of output units

    X : numpy.ndarray
        training data sets
    
    y : numpy.ndarray
        training data sets labels

    lambda_val : double
        regularization parameter

    Returns
    -------
    cost_J : double
        cost function value

    grad : numpy.ndarray
        gradient of the cost function / partial derivative
    """
    np.set_printoptions()

    # unroll the parameters
    [theta_1, theta_2] = roll_params(
        weight_params, input_layer_size, hidden_layer_size, num_labels)

    """
    Forward Propagation Algorithm

    to compute the hypothesis for all examples training data set, the hypotheses
    can be used later on to compute cost function

    also by computing the hypotheses we obviously already computed the activation 
    function for layer 2 and 3 (hidden and output layers). We need this value 
    later on when we are implementing backpropagation

    detailed mathematical representation is available in course website.
    This is the vectorized representation.
    """

    # Calculate the forward propagation
    # Already working properly
    cost_J = -1
    m = X[:, 0].size             # get the size of input layers
    ones = np.ones((m, 1))       # vector of one with the length of m

    # calculate the activation function for the hidden layer
    a1 = np.hstack((ones, X)) # add the bias unit/node
    theta_1_T = np.transpose(theta_1)
    z1 = np.dot(a1, theta_1_T)

    a2 = sigmoid(z1)

    # calculate the hypothesis for the output layer
    a2 = np.hstack((ones, a2)) # add the bias unit/node
    theta_2_T = np.transpose(theta_2)
    z2 = np.dot(a2, theta_2_T)

    a3 = sigmoid(z2)

    y_matrix = y.reshape(-1)
    y_matrix = np.eye(num_labels)[y_matrix-1]  # y-1
    # calculate the cost function
    cost_J = (-1/m) * np.sum(np.sum(y_matrix *
                                    np.log(a3) + (1 - y_matrix) * np.log(1 - a3)))

    theta_1_reg = theta_1[:, 1:theta_1[1, :].size]
    theta_2_reg = theta_2[:, 1:theta_1[1, :].size]

    # calculate the cost function WITH regularization
    cost_J = cost_J + lambda_val / \
        (2*m) * (np.sum(np.sum(theta_1_reg**2)) + np.sum(np.sum(theta_2_reg**2)))

    """
    Backpropagation Algorithm:

    To compute the error for each layer which can be used later on
    to calculate the gradient/partial derivative of the cost function

    detailed mathematical representation is available in course website.
    This is the vectorized representation.
    """

    # initialize the matrix to store the error
    delta_1 = np.zeros(theta_1.shape)
    delta_2 = np.zeros(theta_2.shape)

    for i in range(m):

        a1_i = np.array([a1[i, :]])
        a2_i = np.array([a2[i, :]])
        a3_i = np.array([a3[i, :]])

        y_i = np.array([y_matrix[i, :]])

        error3_i = a3_i - y_i  # Dimension: 1 x num_labels

        # this is to get the z2 of i-th training example
        z2_i = np.dot(theta_1, np.transpose(a1_i))
        # this is to add the bias unit on the second layer
        z2_i = np.vstack([[1], z2_i])

        error2_i = np.dot(np.transpose(theta_2), np.transpose(
            error3_i)) * sigmoid_gradient(z2_i)

        # accumulate the gradient
        delta_1 = delta_1 + np.dot(error2_i[1:error2_i.size], a1_i)
        delta_2 = delta_2 + np.dot(np.transpose(error3_i), a2_i)
    pass

    # calculate the gradient for the neural network cost function (with regularization)
    theta_1_gradient = (1/m) * delta_1 + (lambda_val/m)*np.zeros(theta_1.shape)
    theta_2_gradient = (1/m) * delta_2 + (lambda_val/m)*np.zeros(theta_2.shape)

    # unroll the parameter and concatenate them together.
    grad = unroll_params(theta_1_gradient, theta_2_gradient)

    return cost_J, grad


def predict(theta_1, theta_2, X):
    """Predict the label of a training data set, given a trained neural network weight parameter

    Parameters
    ----------

    theta_1 : numpy.ndarray
        trained matrix of weight that control the function mapping from input layer to hidden layer

    theta_2 : numpy.ndarray
        trained matrix of weight that control the function mapping from hidden layer to output layer

    X : numpy.ndarray

    Returns
    -------
    p : return the prediction of the trained neural network

    """
    m = X.shape[0]
    num_labels = theta_2.shape[0]

    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), theta_1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), theta_2.T))
    p = np.argmax(h2, axis=1)
    return p


def unroll_params(theta_1, theta_2):
    """To unroll two matrix into one long vector by concatenating them together

    Parameters
    ----------

    theta_1 : numpy.ndarray
        matrix of weight that control the function mapping from input layer to hidden layer

    theta_2 : numpy.ndarray
        matrix of weight that control the function mapping from hidden layer to output layer

    Returns
    -------
    thetas : unrolled version of both matrices of weight

    """

    theta_1_flat = theta_1.flatten().reshape(-1, 1)
    theta_2_flat = theta_2.flatten().reshape(-1, 1)
    thetas = np.concatenate((theta_1_flat, theta_2_flat))
    return thetas


def roll_params(thetas, input_layer, hidden_layer, num_labels):
    """To get two matrices that is rolled and concatenated into one long vector

    Parameters
    ----------

    thetas : numpy.ndarray
        an unrolled matrices of weight that consists of theta_1 and theta_2

    input_layer_size : int
        the size of the input layer / the size of input units

    hidden_layer_size : int
        the size of the hidden layer / the number of hidden units

    num_labels : int
        the number of labels this represent digit 0 to 9 / the number of output units

    Returns
    -------
    theta_1 : numpy.ndarray
        matrix of weight that control the function mapping from input layer to hidden layer

    theta_2 : numpy.ndarray
        matrix of weight that control the function mapping from hidden layer to output layer

    """
    separate_point = (input_layer+1) * hidden_layer
    theta_1 = thetas[0:separate_point].reshape(hidden_layer, input_layer+1)
    theta_2 = thetas[separate_point:].reshape(num_labels, hidden_layer+1)
    return theta_1, theta_2
