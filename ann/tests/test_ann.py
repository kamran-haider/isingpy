from ann.ann import Network
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


def test_network_init():
    """
    Test if the network parameter are initialized correctly.
    """

    test_params = {'W1': np.array([[ 0.01788628,  0.0043651 ,  0.00096497, -0.01863493, -0.00277388],
       [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
       [-0.01313865,  0.00884622,  0.00881318,  0.01709573,  0.00050034],
       [-0.00404677, -0.0054536 , -0.01546477,  0.00982367, -0.01101068]]),
    'b1': np.array([[ 0.],
       [ 0.],
       [ 0.],
       [ 0.]]), 
    'W2': np.array([[-0.01185047, -0.0020565 ,  0.01486148,  0.00236716],
       [-0.01023785, -0.00712993,  0.00625245, -0.00160513],
       [-0.00768836, -0.00230031,  0.00745056,  0.01976111]]), 
    'b2': np.array([[ 0.],
       [ 0.],
       [ 0.]])}
    np.random.seed(3)
    nn = Network([5, 4, 3])
    for k in nn.parameters.keys():
        assert_array_almost_equal(nn.parameters[k], test_params[k], decimal=8)


def test_weighted_inputs():
    """
    Test if weighted inputs for a layer are calculated correctly.
    """

    test_Z = np.array([[ 3.26295337, -1.23429987]])
    test_cache = (np.array([[ 1.62434536, -0.61175641], [-0.52817175, -1.07296862], [ 0.86540763, -2.3015387 ]]), 
        np.array([[ 1.74481176, -0.7612069 ,  0.3190391 ]]), 
        np.array([[-0.24937038]]))
    np.random.seed(1)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    nn = Network([])
    Z, cache = nn.weighted_inputs(A, W, b)
    assert_array_almost_equal(Z, test_Z, decimal=8)
    for index, c in enumerate(cache):
        assert_array_almost_equal(c, test_cache[index], decimal=8)


def test_activations():
    """
    Tests if activations for a layer are calculated correctly.
    """

    test_a_sigmoid = np.array([[0.96890023, 0.11013289]])
    test_a_relu = [[3.43896131, 0.]]
    test_sigmoid_cache = ((np.array([[-0.41675785, -0.05626683], [-2.1361961, 1.64027081], [-1.79343559, -0.84174737]]),
            np.array([[0.50288142, -1.24528809, -1.05795222]]),
            np.array([[-0.90900761]])),
            np.array([[3.43896131, -2.08938436]]))
    test_relu_cache = ((np.array([[-0.41675785, -0.05626683],[-2.1361961, 1.64027081],[-1.79343559, -0.84174737]]),
            np.array([[0.50288142, -1.24528809, -1.05795222]]),
            np.array([[-0.90900761]])),
            np.array([[3.43896131, -2.08938436]]))

    np.random.seed(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    nn = Network([])
    A, linear_activation_cache = nn.activations(A_prev, W, b, "sigmoid")
    assert_array_almost_equal(A, test_a_sigmoid, decimal=8)

    c1, c2 = linear_activation_cache
    for index, c in enumerate(c1):
        assert_array_almost_equal(c, test_sigmoid_cache[0][index], decimal=8)

    assert_array_almost_equal(c2, test_sigmoid_cache[1], decimal=8)

    A, linear_activation_cache = nn.activations(A_prev, W, b, "relu")
    assert_array_almost_equal(A, test_a_relu, decimal=8)

    c1, c2 = linear_activation_cache
    for index, c in enumerate(c1):
        assert_array_almost_equal(c, test_relu_cache[0][index], decimal=8)

    assert_array_almost_equal(c2, test_relu_cache[1], decimal=8)


def test_forward_propagation():
    """
    Tests if forward propagation works and network output is correctly obtained.
    """

    test_AL = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    test_cache_size = 3

    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    nn = Network([4, 3, 1])
    # overwrite params with the test ones
    nn.parameters = parameters
    AL, caches = nn.forward_propagation(X)
    assert_array_almost_equal(AL, test_AL, decimal=8)
    np.testing.assert_(test_cache_size == len(caches))


def test_compute_cost():
    """
    Tests if cross-entropy cost is calculated correctly.
    """
    test_cost = 0.414931599615
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8, .9, 0.4]])
    nn = Network([])
    cost = nn.compute_cost(aL, Y)
    np.testing.assert_almost_equal(cost, test_cost)


def test_linear_backward():
    """
    Tests if the linear part of the backward propagation is calculated correctly.
    """
    test_dA_prev = np.array([[0.51822968, -0.19517421], [-0.40506361, 0.15255393], [2.37496825, -0.89445391]])
    test_dW = np.array([[-0.10076895, 1.40685096, 1.64992505]])
    test_db = np.array([[0.50629448]])
    np.random.seed(1)
    dZ = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    linear_cache = (A, W, b)
    nn = Network([])
    dA_prev, dW, db = nn.linear_backward(dZ, linear_cache)
    assert_array_almost_equal(dA_prev, test_dA_prev, decimal=8)
    assert_array_almost_equal(dW, test_dW, decimal=8)
    assert_array_almost_equal(db, test_db, decimal=8)


def test_linear_activation_backward():
    """
    Tests to check if gradients are correctly calculated.
    """
    test_dA_prev_sigmoid = np.array([[ 0.11017994, 0.01105339], [0.09466817, 0.00949723], [-0.05743092, -0.00576154]])
    test_dW_sigmoid = np.array([[0.10266786, 0.09778551, -0.01968084]])
    test_db_sigmoid = np.array([[-0.05729622]])
    test_dA_prev_relu = np.array([[0.44090989, 0.0], [0.37883606, 0.0], [-0.22982280, 0.0]])
    test_dW_relu = np.array([[0.44513824, 0.37371418, -0.10478989]])
    test_db_relu = np.array([[-0.20837892]])

    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)
    nn = Network([])
    dA_prev_sigmoid, dW_sigmoid, db_sigmoid = nn.linear_activation_backward(dA, linear_activation_cache, activation="sigmoid")
    dA_prev_relu, dW_relu, db_relu = nn.linear_activation_backward(dA, linear_activation_cache, activation = "relu")
    assert_array_almost_equal(dA_prev_sigmoid, test_dA_prev_sigmoid, decimal=8)
    assert_array_almost_equal(dW_sigmoid, test_dW_sigmoid, decimal=8)
    assert_array_almost_equal(db_sigmoid, test_db_sigmoid, decimal=8)
    assert_array_almost_equal(dA_prev_relu, test_dA_prev_relu, decimal=8)
    assert_array_almost_equal(dW_relu, test_dW_relu, decimal=8)
    assert_array_almost_equal(db_relu, test_db_relu, decimal=8)

def test_backward_propagation():
    """Tests if gradients are correctly calculated for L-layer model.
    """ 
    test_dW1 = np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                [ 0.0, 0.0, 0.0, 0.0],
                [ 0.05283652, 0.01005865, 0.01777766, 0.0135308]])
    test_db1 = np.array([[-0.22007063], [ 0.0], [-0.02835349]])
    test_dA1 = np.array([[ 0.12913162, -0.44014127], [-0.14175655, 0.48317296], [ 0.01663708, -0.05670698]])

    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    nn = Network([])
    caches = (linear_cache_activation_1, linear_cache_activation_2)
    grads = nn.backward_propagation(AL, Y, caches)

    assert_array_almost_equal(grads["dW1"], test_dW1, decimal=8)
    assert_array_almost_equal(grads["db1"], test_db1, decimal=8)
    assert_array_almost_equal(grads["dA2"], test_dA1, decimal=8)

def test_update_params():
    """
    Tests if gradient descent update of parameters is implemented correctly.
    """
    test_W1 = np.array([[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
                [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
                [-1.0535704, -0.86128581, 0.68284052, 2.20374577]])
    test_b1 = np.array([[-0.04659241], [-1.28888275], [ 0.53405496]])
    test_W2 = np.array([[-0.55569196, 0.0354055, 1.32964895]])
    test_b2 = np.array([[-0.84610769]])

    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    nn = Network([])
    updated_parameters = nn.update_parameters(parameters, grads, 0.1)
    assert_array_almost_equal(updated_parameters["W1"], test_W1, decimal=8)
    assert_array_almost_equal(updated_parameters["b1"], test_b1, decimal=8)
    assert_array_almost_equal(updated_parameters["W2"], test_W2, decimal=8)
    assert_array_almost_equal(updated_parameters["b2"], test_b2, decimal=8)


