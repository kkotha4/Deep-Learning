import numpy as np


class NeuralNetwork:
    """
    A multi-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices.

    The network uses a nonlinearity after each fully connected layer except for the
    last. You will implement two different non-linearities and try them out: Relu
    and sigmoid.

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_sizes, output_size, num_layers, nonlinearity='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H_1)
        b1: First layer biases; has shape (H_1,)
        .
        .
        Wk: k-th layer weights; has shape (H_{k-1}, C)
        bk: k-th layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: List [H1,..., Hk] with the number of neurons Hi in the hidden layer i.
        - output_size: The number of classes C.
        - num_layers: Number of fully connected layers in the neural network.
        - nonlinearity: Either relu or sigmoid
        """
        self.num_layers = num_layers

        assert(len(hidden_sizes)==(num_layers-1))
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params['W' + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params['b' + str(i)] = np.zeros(sizes[i])

        if nonlinearity == 'sigmoid':
            self.nonlinear = sigmoid
            self.nonlinear_grad = sigmoid_grad
        elif nonlinearity == 'relu':
            self.nonlinear = relu
            self.nonlinear_grad = relu_grad


    def forward(self, X):
        """
        Compute the scores for each class for all of the data samples.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.

        Returns:
        - scores: Matrix of shape (N, C) where scores[i, c] is the score for class
            c on input X[i] outputted from the last layer of your network.
        - layer_output: Dictionary containing output of each layer BEFORE
            nonlinear activation. You will need these outputs for the backprop
            algorithm. You should set layer_output[i] to be the output of layer i.

        """
        layer_output = {}
        if self.num_layers==2:


            z_1=(X.dot(self.params["W1"])+self.params["b1"])


            a_1=self.nonlinear(z_1)

            Z_2=a_1.dot(self.params["W2"])+self.params["b2"]
            layer_output["z1"]=z_1
            layer_output["a1"]=a_1
            scores = Z_2



        if self.num_layers==3:
            z_1=(X.dot(self.params["W1"])+self.params["b1"])


            a_1=self.nonlinear(z_1)

            z_2=a_1.dot(self.params["W2"])+self.params["b2"]

            a_2=self.nonlinear(z_2)

            z_3=a_2.dot(self.params["W3"])+self.params["b3"]

            layer_output["z1"]=z_1

            layer_output["z2"]=z_2

            layer_output["a1"]=a_1

            layer_output["a2"]=a_2

            scores = z_3




        #############################################################################
        # TODO: Write the forward pass, computing the class scores for the input.   #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C). Store the output of each layer BEFORE nonlinear activation  #
        # in the layer_output dictionary                                            #
        #############################################################################


        return scores, layer_output


    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """

        # Compute the forward pass
        # Store the result in the scores variable, which should be an array of shape (N, C).
        scores, layer_output = self.forward(X)

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = 0
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss using the scores      #
        # output from the forward function. The loss include both the data loss and #
        # L2 regularization for weights W1,...,Wk. Store the result in the variable #
        # loss, which should be a scalar. Use the Softmax classifier loss.          #
        #############################################################################

        scores=(scores.T-(np.max(scores,axis=1))).T
        scores=np.exp(scores)
        scores = scores / np.sum(scores, axis=1, keepdims=True)
        prob = -np.log(scores[range(len(X)), y])
        loss = np.sum(prob) /len(X)
        '''gradent of softmax loss of output layer'''
        gradient_out = scores
        gradient_out[range(len(X)),y]=gradient_out[range(len(X)),y]-1

        gradient_out = gradient_out/len(X)
        #print(scores.shape)
        '''for i in range(len(scores)):
          sum_v=np.sum(scores[i])
          loss+=(-np.log(scores[i,y[i]]/sum_v))
          scores[i]=scores[i]/sum_v
          scores[i,y[i]]-=1'''

        #scores=scores/len(X)
        #print(scores.shape)
        if self.num_layers==2:

          total_loss=loss+(np.sum(np.square(self.params["W1"]))+np.sum(np.square(self.params["W2"])))*(reg/(2))

        if self.num_layers==3:
           total_loss=loss+(np.sum(np.square(self.params["W1"]))+np.sum(np.square(self.params["W2"]))+np.sum(np.square(self.params["W3"])))*(reg/(2))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        if self.num_layers==2:
          d_z2=gradient_out
          grads['W2'] = np.dot(layer_output["a1"].T,d_z2)
          grads['b2'] = np.sum(d_z2, axis=0)
          d_a1= np.dot(d_z2, self.params["W2"].T)
          d_z1= np.multiply(d_a1,self.nonlinear_grad(layer_output["z1"]))
          grads['W1'] = np.dot(X.T, d_z1)
          grads['b1'] = np.sum(d_z1, axis=0)
          grads['W2'] += reg * self.params["W2"]
          grads['W1'] += reg * self.params["W1"]

        if self.num_layers==3:
          d_z3=gradient_out
          grads['W3'] = np.dot(layer_output["a2"].T,d_z3)
          grads['b3'] = np.sum(d_z3, axis=0)
          d_a2 = np.dot(d_z3, self.params["W3"].T)
          d_z2= np.multiply(d_a2,self.nonlinear_grad(layer_output["z2"]))
          grads['W2'] = np.dot(layer_output["a1"].T,d_z2)
          grads['b2'] = np.sum(d_z2, axis=0)
          d_a1 = np.dot(d_z2, self.params["W2"].T)
          d_z1= np.multiply(d_a1,self.nonlinear_grad(layer_output["z1"]))
          grads['W1'] = np.dot(X.T, d_z1)
          grads['b1'] = np.sum(d_z1, axis=0)
          grads['W2'] += reg * self.params["W2"]
          grads['W1'] += reg * self.params["W1"]
          grads['W3'] += reg * self.params["W3"]

        '''#d_z1=d_a_output.dot(self.nonlinear_grad(layer_output["z1"]))
        d_w2=((d_z2.dot(layer_output["a1"].T)).T)/len(X)+(reg/len(X))*self.params["W2"]
        #print(d_w2.shape)
        d_b2=np.squeeze((np.sum(d_z2,axis=1,keepdims=True))/len(X))
        #print(d_b2.shape)
        d_a1=self.params["W2"].dot(d_z2)
        d_z1=d_a1*(self.nonlinear_grad(layer_output["z1"]))
        d_w1=((d_z1.dot(X)).T)/len(X)+(reg/len(X))*self.params["W1"]
        d_b1=np.squeeze((np.sum(d_z1,axis=1,keepdims=True))/len(X))
        #print(d_b1.shape)
        grads["W2"]=d_w2
        grads["W1"]=d_w1
        grads["b2"]=d_b2
        grads["b1"]=d_b1'''





        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return total_loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        length_data = np.arange(len(X))
        np.random.shuffle(length_data)
        X = X[length_data]
        y = y[length_data]



        for it in range(num_iters):
            sample_indices = np.random.choice(np.arange(len(X)), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            ######

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            if self.num_layers==2:

              self.params["W1"]+=(-learning_rate*grads["W1"])
              self.params["W2"]+=(-learning_rate*grads["W2"])
              self.params["b1"]+=(-learning_rate*grads["b1"])
              self.params["b2"]+=(-learning_rate*grads["b2"])
            if self.num_layers==3:

              self.params["W1"]+=(-learning_rate*grads["W1"])
              self.params["W2"]+=(-learning_rate*grads["W2"])
              self.params["b1"]+=(-learning_rate*grads["b1"])
              self.params["b2"]+=(-learning_rate*grads["b2"])
              self.params["W3"]+=(-learning_rate*grads["W3"])
              self.params["b3"]+=(-learning_rate*grads["b3"])

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        if self.num_layers==2:
            z_1=(X.dot(self.params["W1"])+self.params["b1"].T).T

            a_1=self.nonlinear(z_1)
            Z_2=a_1.T.dot(self.params["W2"])+self.params["b2"]


            y_pred = np.argmax(Z_2, axis=1)

        if self.num_layers==3:
            z_1=(X.dot(self.params["W1"])+self.params["b1"])


            a_1=self.nonlinear(z_1)

            z_2=a_1.dot(self.params["W2"])+self.params["b2"]

            a_2=self.nonlinear(z_2)

            z_3=a_2.dot(self.params["W3"])+self.params["b3"]

            y_pred = np.argmax(z_3, axis=1)


        ###########################################################################
        # TODO: Implement classification prediction. You can use the forward      #
        # function you implemented                                                #
        ###########################################################################

        return y_pred


def sigmoid(X):


    return 1 /(1 + np.exp((-X)))
    #############################################################################
    # TODO: Write the sigmoid function                                          #
    #############################################################################


def sigmoid_grad(X):
    return sigmoid(X)*(1-sigmoid(X))
    #############################################################################
    # TODO: Write the sigmoid gradient function                                 #
    #############################################################################


def relu(X):

    #X[X<=0]=0
    Z=np.maximum(X,0)
    #############################################################################
    #  TODO: Write the relu function                                            #
    #############################################################################
    return Z

def relu_grad(X):
    #############################################################################
    # TODO: Write the relu gradient function                                    #
    #############################################################################
    X[X<=0]=0
    X[X>0]=1

    return X
