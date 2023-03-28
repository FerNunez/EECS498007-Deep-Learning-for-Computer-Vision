"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional


def hello_linear_classifier():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from linear_classifier.py!")


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# **************************************************#
################## Section 1: SVM ##################
# **************************************************#


def svm_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # W =  ([3073, 10]) i=3073 j=10 
    # w00     w01  ...  w0j       first pixel 0
    # w10     w11  ...  w1j
    # w20     w21  ...  w2j
    #..
    # wi0     wi1  ...  wij       # last pixel 3073
    
    # w                 w
    # predicts  ..     predicts
    # class            class
    # 0                 j
    
    # wj
#    print(dW.shape)  = ([3073, 10])
#    print(X[0].shape) = torch.Size([3073])
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                #######################################################################
                # TODO:                                                               #
                # Compute the gradient of the SVM term of the loss function and store #
                # it on dW. (part 1) Rather than first computing the loss and then    #
                # computing the derivative, it is simple to compute the derivative    #
                # at the same time that the loss is being computed.                   #
                #######################################################################
                # Replace "pass" statement with your code
                # Source: https://cs231n.github.io/optimization-1/
                # Derivate is in 2 parts.. derv wj for j
                
                # For all incorrect labels (col of j != y[i]) its dervi is x
                dW[:, j] = dW[:, j] + X[i].t()
                
                
                # For j==y[i] (wights for correct label), deriv is sum of -X for all j!=y[i])
                dW[:, y[i]] = dW[:, y[i]] - X[i].t()
                # This sum can be made outs side of j loop is we would wanted it
                # like -X[i]*(num_classes-1)
    
    # I guess I should devide by images train to get mean?
    dW = dW/num_train
    # yeahm jus tlike loss is divided
#######################################################################
                #                       END OF YOUR CODE                              #
                #######################################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function w.r.t. the regularization term  #
    # and add it to dW. (part 2)                                                #
    #############################################################################
    # Replace "pass" statement with your code
    
    # L2 regularization add reg *SumSum(W^2)
    
    dW = dW + reg*2*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    #print(W.shape)  #torch.Size([3073, 10])
    #print(X.shape) #torch.Size([128, 3073])
    num_train = X.shape[0]
    
    # score of each 10 clases for 128 classes 
    scores_by_image = (X.matmul(W)) # torch.Size([128, 10])
    #print(scores_by_image.shape) ## torch.Size([128, 10])
    
    #y[i] = c means that X[i] has label c, where 0 <= c < C. (i in num_train)    
    #print(y.shape) ##torch.Size([128]
    # Correct class score for each image:
    correct_class_score_by_image = scores_by_image[range(scores_by_image.shape[0]), y] # #torch.Size([128])
    
    #print(correct_class_score_by_image.shape) #torch.Size([128])
    
    # get margin (sj - sy +1)
    # margin value for each class for all images
    margin_by_class =  scores_by_image.t() - correct_class_score_by_image + 1 
    #print(margin_by_class.shape) #torch.Size([10, 128])
    
    # invert to get 10 classes foro each 128 images: #torch.Size([128, 10])
    margin = margin_by_class.t()

    # Reset margin value for correct class for each image 
    margin[range(margin.shape[0]), y] = 0
    
    # Apply max (0, margin)
    margin[margin<0] = 0

    
    # loss by Sum everything (class and images)
    loss = torch.sum(margin)

#     # substract loss for cases where j==y => then margin gives 1, Num images time
#     loss -= 1*num_train
    
    #devide by number of images:
    loss /= num_train

    # Add regularization
    loss += reg * torch.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # W =  ([3073, 10]) i=3073 j=10 
    # w00     w01  ...  w0j       first pixel 0
    # w10     w11  ...  w1j
    # w20     w21  ...  w2j
    #..
    # wi0     wi1  ...  wij       # last pixel 3073
    
    # w                 w
    # predicts  ..     predicts
    # class            class
    # 0                 j
        
    # Convert into Indicator function
    margin[margin>0] = 1
    
    
    # margin #torch.Size([128, 10])  
    # 
    # 
    # m00 m01 m02 ... m0j  # image0
    # m10 m11 m12     m1j  # image1
    # m20 m21 m22     m2j
    # m30 m31 m32 ...
    # ...
    # mi0 mi1 mi2     mij  # imagei
    #
    # cl  cl  cl      cl
    # ass ass ass     ass
    # 0   1   2       i
    
    # checkk filas contra columnas
    #∇wyiLi=−(∑j≠yi1(wTjxi−wTyixi+Δ>0))xi
    
    # derivative resect to Wy (correct class)
    # sum of Indicator function * -X
    correct_label_vals = -torch.sum(margin, dim=1)
#     print("qsd")
#     print(correct_label_vals.shape)   #torch.Size([128])
    
    margin[range(margin.shape[0]), y] = correct_label_vals
    
    # mm() matrix multi
    dW = X.t().mm(margin)
    
#     print(X.t().shape) #torch.Size([3073, 128])
#     print(margin.shape) #torch.Size([128, 10])
    dW /= num_train
    
    dW = dW + reg*2*W
    #     correct_label_vals = -torch.sum(margin_by_class , dim=1) 
#     margin_by_class[range(margin_by_class.shape[0]), y.t()] = correct_label_vals 
    
    
#     dW += X.t().mm(margin_by_class)
    
#     dW /= num_train
    
#     dW += reg * 2* W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    #########################################################################
    # TODO: Store the data in X_batch and their corresponding labels in     #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
    # and y_batch should have shape (batch_size,)                           #
    #                                                                       #
    # Hint: Use torch.randint to generate indices.                          #
    #########################################################################
    # Replace "pass" statement with your code
    # torch.randint(low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

    
    random_indexes = torch.randint(low=0, high=num_train, size=(batch_size, ))
    X_batch = X[random_indexes]
    y_batch = y[random_indexes]
    
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return X_batch, y_batch


def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        # DONE: implement sample_batch function
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # perform parameter update
        #########################################################################
        # TODO:                                                                 #
        # Update the weights using the gradient and the learning rate.          #
        #########################################################################
        # Replace "pass" statement with your code
        W -=  learning_rate * grad 
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # Replace "pass" statement with your code
    scores = X.mm(W) #torch.Size([40000, 10])
    #gives a N,C
    (values, y_pred) = torch.max(scores, dim=1)
    
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """

    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO:   add your own hyper parameter lists.                             #
    ###########################################################################
    # Replace "pass" statement with your code
    learning_rates = [3.3e-3, 1e-2, 3.3e-2, 1e-1, 3.3e-1]
    regularization_strengths = [ 3.3e-4, 1e-3, 3.3e-3, 1e-2, 3.3e-2]
 
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.
    ###########################################################################
    # TODO:                                                                   #
    # Write code that, train a linear SVM on the training set, compute its    #
    # accuracy on the training and validation sets                            #
    #                                                                         #
    # Hint: Once you are confident that your validation code works, you       #
    # should rerun the validation code with the final value for num_iters.    #
    # Before that, please test with small num_iters first                     #
    ###########################################################################
    # Feel free to uncomment this, at the very beginning,
    # and don't forget to remove this line before submitting your final version
#     num_iters = 100
    
    # Replace "pass" statement with your code
    cls.train(data_dict['X_train'], data_dict['y_train'], lr, reg,
        num_iters)
    
    y_train_pred = cls.predict(data_dict['X_train'])
    y_val_pred = cls.predict(data_dict['X_val'])

    train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).double().mean().item()
    val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).double().mean().item()

    
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################

    return cls, train_acc, val_acc


# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, naive implementation (with loops).  When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    
    #print(W.shape) #torch.Size([3073, 10])
    #print(X.shape) #torch.Size([128, 3073])
    #print(y.shape) #torch.Size([128])

    num_train = X.shape[0]
    num_class = W.shape[1]
    
    for i in range(0,num_train):        
        logits = W.t().mv(X[i])
        maxi = torch.max(logits) 
        logits -= maxi
        unnorm_prob = torch.exp(logits)
        sum_prob = torch.sum(unnorm_prob)
        prob = unnorm_prob/sum_prob
        loss += -torch.log(prob[y[i]])
        
        for j in range(num_class):
            dW[:,j] += X[i]* (prob[j] - 1*(j == y[i]) )
    
    
    # dLi/dfi = -1 + e^fj/sum (e^fk) = -1+pj, if (j=yi)  
    # dLi/dfi = pj (j=!1)
    # dLi/dfi = pj - 1*(j == p[y])
    
    # dfi/dW = Xi
    
    
    # get media
    loss /= num_train
    dW /= num_train
    
    # Apply regularization
    loss += reg * torch.sum(W * W)
    dW += reg*2*W
 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    num_train = X.shape[0]

    # get logits
    logits = X.mm(W)
    max_value,_ = torch.max(logits, dim=1, keepdim=True)

    logits -=max_value   
    #print(logits.shape)  #torch.Size([128, 10])
    
    # unnormalized probabilities
    unnorm_prob = torch.exp(logits)
    #print(unnorm_prob.shape)  #torch.Size([128, 10])
    
    # Get sum of all unnorm proba for classes for each image
    sum_prob = torch.sum(unnorm_prob, dim=1, keepdims=True)
    #print(sum_prob.shape) #torch.Size([128])
    
    # Normalize probabilities
    prob = unnorm_prob / sum_prob
    #print(prob.shape) #torch.Size([128, 10])
    
    # Compute Loss and get mean
    L_vector = -torch.log(prob[range(prob.shape[0]), y])
    #print(L_vector.shape) #torch.Size([128])
    loss = torch.sum(L_vector)/X.shape[0]
    
    

    
    #https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    #dL/dW = dz/dW * dL/dz
    
    #dz/dW = X cause z = w*x
    #dL/dz = s_i - y_i where s_i = e^z_i / sum(e^z_l) and y_i = 1 if correct label 
    # prob - Substract -1 toto correct class by image
    prob[range(prob.shape[0]), y] -=  1
    #print(prob.shape) #torch.Size([128, 10])
    
    #print(X.shape) #torch.Size([128, 3073])
    # why mult by input? dont get it
    dW = X.t().mm(prob)
    dW /= num_train
    dW += reg*2*W

    #print(dW.shape) #torch.Size([3073, 10])
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    learning_rates = [3.3e-3, 1e-2, 3.3e-2, 1e-1, 3.3e-1]
    regularization_strengths = [ 3.3e-5, 1e-4, 3.3e-4, 1e-3, 3.3e-3]
   
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths
