# Useful function to reconsider:
#   .flatten()
#   .multiply() : elementwise multiplication
#   .eye() : genrate identity matrix
#
import utilsML as util
import numpy as np
from mlxtend.data import loadlocal_mnist
from scipy.io import loadmat
from scipy import optimize


def main():
    np.set_printoptions(precision=2, formatter={
        'float': lambda x: format(x, '6.3E')})

    # change these variables to do a different testing
    TEST_RUN = 1

    DISPLAY_DATA = 0
    DATA_SET = 0
    LAMBDA_ = 1
    HU = 25
    ITER = 100
    PROMPT = 0

    SAMPLE = 5000

    TRAINING_SAMPLE = int(0.8 * SAMPLE)
    TEST_SAMPLE = int(0.2 * SAMPLE)
 
    total = 0

    print("Data set: ", DATA_SET)
    print("Training sample: ", TRAINING_SAMPLE)
    print("Test sample: ", TEST_SAMPLE)
    print("Lambda (Regularization): ", LAMBDA_)
    print("Hidden units: ", HU)
    print("Training iteration: ", ITER)

    if DISPLAY_DATA == 1:
        disp_sample = np.random.randint(5000, size=10)
        # X, y = loadlocal_mnist(
        #         images_path='./train-images-idx3-ubyte',
        #         labels_path='./train-labels-idx1-ubyte')
        # util.display_digit(X[disp_sample], 28)

        load_mat = loadmat('ex4data1.mat')
        X = load_mat['X']
        y = load_mat['y']
        util.display_digit(X[disp_sample], 20)

    for t in range(0, TEST_RUN):
        acc = test_model(DATA_SET, TRAINING_SAMPLE, TEST_SAMPLE, LAMBDA_, HU, ITER, PROMPT)
        total = total + acc
        # print("test %d, accuracy: %f" % (t, acc))
        print(acc)
        pass
    
    print("Final accuracy: %f" % (total/TEST_RUN))

    # reached end of program.
    print("\n====================================")
    print("End of Program.")
    print("Devin Efendy, COMP3190, Assignment 4")
    pass


def test_model(data_set, sample_size, test_size, lambda_val, hidden_layer_sz, maxiter, PROMPT):


    # variables for neural networks layers
    # this is the input layer, layer that we feed our training data sets
    input_layer_size = 784  # 28x28 input images flattened to 784 pixels
    # the hidden/midden layer of our neural networks
    hidden_layer_size = 25  # original is 25
    # the number of output layers
    num_labels = 10        # the output layers will be from 0 - 9

    # loading data sets (60000 x 784)
    # e.g. 60000 row(training datas) and 784 column(features, pixel intensity)
    if PROMPT == 1:
        print("\n********************************* START TEST *********************************")
        print("Loading and preparing data sets...")
        print("--------------------------------------------------------------------")
        pass

    DATA_SET = data_set
    USE_RANDOM_SAMPLE = 1
    RANDOM_SAMPLE = sample_size
    TEST_SAMPLE = test_size
    lambda_ = lambda_val
    iteration = maxiter

    X = None
    y = None
    X_sample = None
    y_sample = None
    X_test = None
    y_test = None

    input_layer_size = 0
    hidden_layer_size = hidden_layer_sz
    num_labels = 10

    if DATA_SET == 0:
        load_mat = loadmat('ex4data1.mat')
        X = load_mat['X']
        y = load_mat['y']

        # test_sample = np.random.randint(X.shape[0], size=TEST_SAMPLE)
        # [900 300 500 1 10 ...]
        # scalar = StandardScaler()
        # X = scalar.fit_transform(X)

        training_sample = np.random.choice(X.shape[0], RANDOM_SAMPLE, replace=False)
        # test_sample = np.random.randint(X.shape[0], size=TEST_SAMPLE)

        if RANDOM_SAMPLE == X.shape[0]:
            test_sample = np.arange(X.shape[0])
        else:
            test_sample = np.setdiff1d(np.arange(X.shape[0]), training_sample)

        test_sample = np.random.choice(test_sample, TEST_SAMPLE, replace=False)

        if RANDOM_SAMPLE == TEST_SAMPLE:
            training_sample = np.arange(X.shape[0])
            test_sample =  np.arange(X.shape[0])

        X_test = X[test_sample]
        y_test = y[test_sample]

        X_sample = X[training_sample]
        y_sample = y[training_sample]
        pass
    else:
        X, y = loadlocal_mnist(
            images_path='./train-images-idx3-ubyte',
            labels_path='./train-labels-idx1-ubyte')
        y = y+1
        X_test, y_test = loadlocal_mnist(
            images_path='./t10k-images-idx3-ubyte',
            labels_path='./t10k-labels-idx1-ubyte'
        )

        test_sample = np.random.choice(X_test.shape[0], TEST_SAMPLE, replace=False)
        # test_sample = np.random.randint(X.shape[0], size=TEST_SAMPLE)
        training_sample = np.random.choice(X.shape[0], RANDOM_SAMPLE, replace=False)

        X_test = X_test[test_sample]
        y_test = y_test + 1
        y_test = y_test[test_sample]

        X_sample = X[training_sample]
        y_sample = y[training_sample]
        pass

    # unique, counts = np.unique(y, return_counts=True)
    # print(dict(zip(unique, counts)))

    # unique, counts = np.unique(y_sample, return_counts=True)
    # print(dict(zip(unique, counts)))

    # unique, counts = np.unique(y_test, return_counts=True)
    # print(dict(zip(unique, counts)))


    # get the number of training data sets
    m = X_sample.shape[0]

    input_layer_size = X_sample.shape[1]  # X_sample[1, :].size
    pix_dim = (np.sqrt(input_layer_size)).astype(int)
    if PROMPT == 1:
        print("Shape of training data sets, X:", X_sample.shape)
        print("Number of sample training data sets, m:", m)
        print("Number of Original training data sets, m:", X.shape[0])
        print("Number of features, %dx%d pixels intensity: %d" % (pix_dim, pix_dim, input_layer_size))
        pass

    """
    Random initialization of weight matrices, theta
    """

    initial_theta_1 = util.rand_weight_init(input_layer_size, hidden_layer_size)
    initial_theta_2 = util.rand_weight_init(hidden_layer_size, num_labels)

    weight_params = util.unroll_params(initial_theta_1, initial_theta_2)

    """
    Training Neural Networks:
    """
    if PROMPT == 1:
        print("\nTraining Neural Networks...")
        print("--------------------------------------------------------------------")
        pass

    min_cost_function = lambda weights: util.cost_function(weights,
                                                       input_layer_size, hidden_layer_size, num_labels,
                                                       X_sample, y_sample,
                                                       lambda_)

    options = {'maxiter': iteration}
    # res = util.train_neural_nets(cost_function, weight_params, options)
    result = optimize.minimize(min_cost_function, weight_params, jac=True, method='TNC', options=options)
    # print(result.x)
    # print("Sum: %f", np.sum(result.x))

    [theta_1, theta_2] = util.roll_params(result.x, input_layer_size, hidden_layer_size, num_labels)

    """
    Check the accuracy of ML model
    """

    pred = util.predict(theta_1, theta_2, X_test)

    hit = 0
    miss = 0

    for i in range(TEST_SAMPLE):
        diff = np.floor(np.abs(pred[i]+1 - y_test[i]))
        if pred[i]+1 == y_test[i]:
            hit = hit + 1
        else:
            # print("%d-th sample= pred: %f, y: %f, diff: %f"%(i, pred[i]+1, y[i], pred[i]+1 - y[i]))
            miss = miss + 1

    # print((hit/m) * 100)
    # print((miss/m) * 100)

    accuracy = (hit/TEST_SAMPLE) * 100

    if PROMPT == 1:
        print("Training Iteration     : ", iteration)
        print("Training data set used : ", DATA_SET)
        if USE_RANDOM_SAMPLE == 1:
            print("Random sample size     : ", RANDOM_SAMPLE)
        print("Test sample size       : ", TEST_SAMPLE)
        print("Lambda (Regularization): ", lambda_)
        print("Number of Input Layer  : ", X.shape[1])
        print("Number of Hidden Layer :  1")
        print("     Hidden layer size : ", hidden_layer_size)
        print("Number of output layer : ", num_labels)
        print('Training Set Accuracy  : %f' % (accuracy))
        # print(m)
        print("\n********************************** TEST END **********************************\n\n")
        pass

    return accuracy

main()