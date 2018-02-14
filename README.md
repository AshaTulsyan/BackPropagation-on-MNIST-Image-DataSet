# Back Propagation on MNIST DataSet
The code implements Backpropagation on a feedforward neural network using Stochastic Gradient Descent for classification on MNIST dataset. Initial experiments were done with vanilla SGD, but the process was optimized by using following 'tricks of the trade' from Yann LeCun's paper on Efficient Backpropagation from 1998.
# Some tricks of the trade used
* Input normalization
* Use of Nesterov Momentum Accelerated Gradient
* Efficient Weights initialization
Inputs were normalized by dividing them by 127.5 and subtracting 1. This was done so that inputs are in the range -1 to 1, centered around 0. Also, the input data was split in 80:20 ratio into training/validation dataset.

Nesterov Momentum Accelerated Gradient is an improvement over simple momentum, in the sense that it makes use of 'Look-Ahead Weights' and calculates gradient using them to update the weights. This speeds up convergence by a huge margin, as was observed in the trials conducted for this task.

Weights were initialized randomly from Normal distribution with mean 0 and variance equal to 1/(fan-in), where fan-in is the number of inputs to a neuron.

The architecture used has 1 hidden layers, 64 neurons in the hidden layer. The hidden layers has tanh activation functions, as specified in the task. The output layer has 10 neurons, each corresponding to 1 digit in MNIST. Softmax activation has been used for output layer, and a cross-entropy loss function has been used to observe model's behavior.

Various experiments were conducted, changing the architecture of network - number of neurons in the hidden layers. Early stopping method was used to determine the epoch at which validation accuracy is maximum, and those weights were used to test the model.

# Libraries used
* NumPy
* mnist (to import MNIST data)
Implementation
The value of learning rate is taken as 0.001, which was chosen by hit-and-trial. And the number of epochs is set to 200. Also, since Nesterov Momentum method is used, the value of momentum factor (alpha) is set to 0.9. This is just an empirical value which is used as it is. The current batch size being used is 256. To run the code, follow these steps:

Make sure the MNIST data files ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte') are present in the same directory as the code.

Make sure that library 'mnist' is installed on the system. Use the command

pip install python-mnist
to install the library

To run the code, use following command
python backprop_nn.py
Results
# For the values of hyperparameters mentioned above, and for the architecture mentioned above, the results obtained were:

Early Stopping Epoch : 412

Training Accuracy : 95.7%

Testing Accuracy : 94.28%

Validation Accuracy : 94.72%

Note: This is nowhere close to state-of-the-art methods, but the code implementation is correct, with additional optimizations implemented that mainly affect the training speed. Proper tuning of the model was not performed due to huge computation time and limited resources with me, but with better architecture, greater accuracies can be achieved. One such method is to use 2 hidden layers with 64 neurons each.
Technicalities and Brief Code Description
The code currently doesn't support inputs from command prompt, since architecture was specified initially. But the code is modular enough that by changing arguments to the FeedForward function invoked in main, one can change the number of neurons in the hidden layer, and activation function. 
# Following are the arguments to Feedforward function and their roles:

Images_train : This takes in the entire training input data (split is performed inside) as a NumPy matrix.

label_train_oh : This takes in the output labels for training data, in one-hot encoded form as a NumPy matrix.

Images_val : This takes in the entire validation input data(20% of the training data) as a NumPy matrix.

label_val_oh : This takes in the output labels for validation data, in one-hot encoded form as a NumPy matrix.

images_test  : This takes in the entire test input data as a NumPy matrix.

label_test_oh :This takes in the output labels for test data, in one-hot encoded form as a NumPy matrix.

H : This is the number of nodes we want at our hidden layer.

l1,l2,l3 : These are the learning rates at the input, hidden, output layer.

epochs : This specifies the number of iterations we will run on the data. Has to be a number ex:200.

shuffle1 : This is the shuffle factor, which is 1 if we want to shuffle our data after every epoch.

momentum : This is the momentum factor (alpha) to be used in case a momentum gradient descent method is decided to be implemented. Default value is 0 i.e. no momentum.

tan : This is the activation function which will be used at the hidden layer.
