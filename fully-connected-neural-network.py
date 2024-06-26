"""
ASHWANTH KUPPUSAMY
"""

from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F

font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Load the data from your Google Drive
data_path = 'cifar_2class_py3.p'


######################################################
# Q1 Implement Init, Forward, and Backward For Layers
######################################################

class SigmoidCrossEntropy:

    # Compute the cross entropy loss after sigmoid. The reason they are put into the same layer is because the gradient has a simpler form
    # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
    # labels -- batch_size x 1 vector of integer label id (0,1) where labels[i] is the label for batch element i
    #
    def __init__(self):
        self.i_logits = None
        self.i_labels = None

    # TODO: Output should be a positive scalar value equal to the average cross entropy loss after sigmoid
    def forward(self, logits, labels):
        self.i_logits = logits
        self.i_labels = labels
        sigmoid_logits = 1 / (1 + np.exp(-self.i_logits))
        # Compute cross entropy loss
        epsilon = 1e-12  # to prevent division by zero and log of zero
        y_pred = np.clip(sigmoid_logits, epsilon, 1. - epsilon)
        return -np.mean(self.i_labels * np.log(y_pred) +
                        (1 - self.i_labels) * np.log(1 - y_pred))
        raise Exception('Student error: You haven\'t implemented the forward pass for SigmoidCrossEntropy yet.')

    # TODO: Compute the gradient of the cross entropy loss with respect to the the input logits
    def backward(self):
        sigmoid_logits = 1 / (1 + np.exp(-self.i_logits))
        grad_logits = (sigmoid_logits - self.i_labels) / self.i_logits.shape[0]
        return grad_logits

        raise Exception('Student error: You haven\'t implemented the backward pass for SigmoidCrossEntropy yet.')


class ReLU:

    # TODO: Compute ReLU(input) element-wise
    def __init__(self):
        self.input = None

    def forward(self, inp):
        self.input = inp
        return np.maximum(0, self.input)
        raise Exception('Student error: You haven\'t implemented the forward pass for ReLU yet.')

    # TODO: Given dL/doutput, return dL/dinput
    def backward(self, grad):
        return grad * (self.input > 0)
        raise Exception('Student error: You haven\'t implemented the backward pass for ReLU yet.')

    # No parameters so nothing to do during a gradient descent step
    def step(self, step_size, momentum=0, weight_decay=0):
        return


class LinearLayer:

    # TODO: Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim):
        self.prev_grad_bias = None
        self.prev_grad_weights = None
        self.input = None
        self.grad_weights = None
        self.grad_bias = None
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros((1, output_dim))
        return
        raise Exception('Student error: You haven\'t implemented the init for LinearLayer yet.')

    # TODO: During the forward pass, we simply compute XW+b
    def forward(self, inp):
        self.input = inp
        return np.dot(self.input, self.weights) + self.bias
        raise Exception('Student error: You haven\'t implemented the forward pass for LinearLayer yet.')

    # TODO: Backward pass inputs:
    def backward(self, grad):
        self.grad_weights = np.dot(self.input.T, grad)
        self.grad_bias = np.sum(grad, axis=0)

        # Compute dL/dX
        grad_input = np.dot(grad, self.weights.T)
        return grad_input
        raise Exception('Student error: You haven\'t implemented the backward pass for LinearLayer yet.')

    ######################################################
    # Q2 Implement SGD with Weight Decay
    ######################################################
    def step(self, step_size, momentum=0.8, weight_decay=0.0):
        grad_weights = None
        grad_bias = None
        if weight_decay > 0:
            self.weights -= weight_decay * self.weights
            self.bias -= weight_decay * self.bias

        # Apply momentum
        if self.prev_grad_weights is not None and self.prev_grad_bias is not None:
            grad_weights = momentum * self.prev_grad_weights + (1 - momentum) * self.grad_weights
            grad_bias = momentum * self.prev_grad_bias + (1 - momentum) * self.grad_bias
            self.weights -= step_size * grad_weights
            self.bias -= step_size * grad_bias
        else:
            # Update the weights and biases
            self.weights -= step_size * self.grad_weights
            self.bias -= step_size * self.grad_bias

        # Store the current gradients for the next step
        self.prev_grad_weights = self.grad_weights
        self.prev_grad_bias = self.grad_bias
        # TODO: Implement the step
        return
        raise Exception('Student error: You haven\'t implemented the step for LinearLayer yet.')


######################################################
# Q4 Implement Evaluation for Monitoring Training
######################################################
# TODO: Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
    # Compute validation loss and accuracy
    val_losses, val_accs = [], []
    # Iterate over data in batches
    for i in range(0, len(X_val), batch_size):
        # Get batch data
        X_batch = X_val[i:i + batch_size]
        Y_batch = Y_val[i:i + batch_size]

        # Forward pass
        val_logits = model.forward(X_batch)

        val_loss = SigmoidCrossEntropy().forward(val_logits, Y_batch)
        val_losses.append(val_loss)

        # Update total loss
        val_predictions = np.where(val_logits < 0, [0], [1])
        val_acc = np.mean(val_predictions == Y_batch)
        val_accs.append(val_acc)

    # Compute average loss and accuracy
    avg_loss = np.mean(val_losses)
    accuracy = np.mean(val_accs)

    return avg_loss, accuracy
    raise Exception('Student error: You haven\'t implemented the step for evaluate function.')


#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):

        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        else:
            self.layers = []
            self.layers.append(LinearLayer(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(ReLU())
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
            self.layers.append(ReLU())
            self.layers.append(LinearLayer(hidden_dim, output_dim))

    # TODO: Please create a network with hidden layers based on the parameters

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size, momentum, weight_decay):
        for layer in self.layers:
            layer.step(step_size, momentum, weight_decay)


def displayExample(x):
    r = x[:1024].reshape(32, 32)
    g = x[1024:2048].reshape(32, 32)
    b = x[2048:].reshape(32, 32)

    plt.imshow(np.stack([r, g, b], axis=2))
    plt.axis('off')
    plt.show()


"""Training the model for different learning rates"""

test_acc_learning_rate = []

learning_rates = [0.1, 0.01, 0.001, 0.0001]

for step_size in learning_rates:
    batch_size = 32
    max_epochs = 100

    number_of_layers = 3
    width_of_layers = 128
    weight_decay = 0.0
    momentum = 0.9

    # Load data
    data = pickle.load(open(data_path, 'rb'))
    X_train = data['train_data']
    Y_train = data['train_labels']
    X_test = data['test_data']
    Y_test = data['test_labels']

    # Some helpful dimensions
    num_examples, input_dim = X_train.shape
    output_dim = 1  # number of class labels

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below. You can edit this as you please.
    net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    # Normalize
    X_train_normalized = X_train / 255
    X_test_normalized = X_test / 255

    for epoch in range(max_epochs):
        for i in range(0, num_examples, batch_size):
            # Get mini-batch
            X_batch = X_train_normalized[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            # Forward pass
            logits = net.forward(X_batch)

            # Compute loss
            sce = SigmoidCrossEntropy()
            loss = sce.forward(logits, Y_batch)
            losses.append(loss)

            # Backward pass
            grad_logits = sce.backward()
            net.backward(grad_logits)

            # Update weights
            net.step(step_size, momentum, weight_decay)

            # Compute accuracy
            predictions = np.where(logits < 0, [0], [1])
            acc = np.mean(predictions == Y_batch)
            accs.append(acc)

        t_loss, t_acc = evaluate(net, X_test_normalized, Y_test, batch_size)
        val_losses.append(t_loss)
        val_accs.append(t_acc)
        print(
            f'Epoch {epoch}, Loss: {np.mean(losses)}, Misclassification rate: {1 - np.mean(accs)},Test Loss: {t_loss}, Test Misclassification rate: {1 - t_acc}')

    test_acc_learning_rate.append(val_accs)

    # Plot training and testing curves
    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))], val_losses, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim(-0.01,3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs, c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    plt.title("Learning rate : " + str(step_size))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.savefig(f'./tuning_results/learning_rate_{step_size}.png')

# Plot for different learning rates
epochs = range(1, len(test_acc_learning_rate[0]) + 1)  # Number of epochs
# Plot each learning rate's accuracies
plt.figure(figsize=(16, 9))
for i, acc in enumerate(test_acc_learning_rate):
    plt.plot(epochs, acc, label=f"Learning Rate: {learning_rates[i]}")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend(loc="lower right")
plt.savefig(f'./tuning_results/learning_rate_test.png')

"""Training the model for different batch sizes"""

test_acc_batch_size = []

batch_sizes = [8, 16, 32, 64, 128]  # Different batch sizes used

for batch_size in batch_sizes:
    max_epochs = 100
    step_size = 0.001
    number_of_layers = 3
    width_of_layers = 128
    weight_decay = 0.0
    momentum = 0.9

    # Load data
    data = pickle.load(open(data_path, 'rb'))
    X_train = data['train_data']
    Y_train = data['train_labels']
    X_test = data['test_data']
    Y_test = data['test_labels']

    # Some helpful dimensions
    num_examples, input_dim = X_train.shape
    output_dim = 1  # number of class labels

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below. You can edit this as you please.
    net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    # Normalize
    X_train_normalized = X_train / 255
    X_test_normalized = X_test / 255

    for epoch in range(max_epochs):
        for i in range(0, num_examples, batch_size):
            # Get mini-batch
            X_batch = X_train_normalized[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            # Forward pass
            logits = net.forward(X_batch)

            # Compute loss
            sce = SigmoidCrossEntropy()
            loss = sce.forward(logits, Y_batch)
            losses.append(loss)

            # Backward pass
            grad_logits = sce.backward()
            net.backward(grad_logits)

            # Update weights
            net.step(step_size, momentum, weight_decay)

            # Compute accuracy
            predictions = np.where(logits < 0, [0], [1])
            acc = np.mean(predictions == Y_batch)
            accs.append(acc)

        t_loss, t_acc = evaluate(net, X_test_normalized, Y_test, batch_size)
        val_losses.append(t_loss)
        val_accs.append(t_acc)
        print(
            f'Epoch {epoch}, Loss: {np.mean(losses)}, Misclassification rate: {1 - np.mean(accs)},Test Loss: {t_loss}, Test Misclassification rate: {1 - t_acc}')

    test_acc_batch_size.append(val_accs)

    # Plot training and testing curves
    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))], val_losses, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim(-0.01,3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs, c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    plt.title(f"Batch size : {batch_size}")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.savefig(f'./tuning_results/batch_size_{batch_size}.png')

# Plot for different batch_sizes
epochs = range(1, len(test_acc_batch_size[0]) + 1)  # Number of epochs
# Plot each learning rate's accuracies
plt.figure(figsize=(16, 9))
for i, acc in enumerate(test_acc_batch_size):
    plt.plot(epochs, acc, label=f"Batch_size: {batch_sizes[i]}")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend(loc="lower right")
plt.savefig('./tuning_results/batch_size_test.png')

"""Training the model for different number of hidden layers"""

test_acc_hidden_unit = []

hidden_units = [1, 2, 3, 4, 5]  # Different batch sizes used

for number_of_layers in hidden_units:
    max_epochs = 100
    step_size = 0.001
    batch_size = 64
    width_of_layers = 128
    weight_decay = 0.0
    momentum = 0.9

    # Load data
    data = pickle.load(open(data_path, 'rb'))
    X_train = data['train_data']
    Y_train = data['train_labels']
    X_test = data['test_data']
    Y_test = data['test_labels']

    # Some helpful dimensions
    num_examples, input_dim = X_train.shape
    output_dim = 1  # number of class labels

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below. You can edit this as you please.
    net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    # Normalize
    X_train_normalized = X_train / 255
    X_test_normalized = X_test / 255

    for epoch in range(max_epochs):
        for i in range(0, num_examples, batch_size):
            # Get mini-batch
            X_batch = X_train_normalized[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            # Forward pass
            logits = net.forward(X_batch)

            # Compute loss
            sce = SigmoidCrossEntropy()
            loss = sce.forward(logits, Y_batch)
            losses.append(loss)

            # Backward pass
            grad_logits = sce.backward()
            net.backward(grad_logits)

            # Update weights
            net.step(step_size, momentum, weight_decay)

            # Compute accuracy
            predictions = np.where(logits < 0, [0], [1])
            acc = np.mean(predictions == Y_batch)
            accs.append(acc)

        t_loss, t_acc = evaluate(net, X_test_normalized, Y_test, batch_size)
        val_losses.append(t_loss)
        val_accs.append(t_acc)
        print(
            f'Epoch {epoch}, Loss: {np.mean(losses)}, Misclassification rate: {1 - np.mean(accs)},Test Loss: {t_loss}, Test Misclassification rate: {1 - t_acc}')

    test_acc_hidden_unit.append(val_accs)

    # Plot training and testing curves
    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))], val_losses, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim(-0.01,3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs, c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    plt.title(f'Hidden Units : {number_of_layers}')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.savefig(f'./tuning_results/hidden_units_{number_of_layers}.png')

# Plot for different learning rates
epochs = range(1, len(test_acc_hidden_unit[0]) + 1)  # Number of epochs
# Plot each learning rate's accuracies
plt.figure(figsize=(16, 9))
for i, acc in enumerate(test_acc_hidden_unit):
    plt.plot(epochs, acc, label=f"Hidden Units: {hidden_units[i]}")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend(loc="lower right")
plt.savefig('./tuning_results/hidden_units_test.png')

"""Training the best model"""

max_epochs = 300
step_size = 0.001
batch_size = 8
width_of_layers = 128
weight_decay = 0.000001
momentum = 0.9
number_of_layers = 3
# Load data
data = pickle.load(open(data_path, 'rb'))
X_train = data['train_data']
Y_train = data['train_labels']
X_test = data['test_data']
Y_test = data['test_labels']

# Some helpful dimensions
num_examples, input_dim = X_train.shape
output_dim = 1  # number of class labels

# Build a network with input feature dimensions, output feature dimension,
# hidden dimension, and number of layers as specified below. You can edit this as you please.
net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

# Some lists for book-keeping for plotting later
losses = []
val_losses = []
accs = []
val_accs = []

# Normalize
X_train_normalized = X_train / 255
X_test_normalized = X_test / 255

for epoch in range(max_epochs):
    for i in range(0, num_examples, batch_size):
        # Get mini-batch
        X_batch = X_train_normalized[i:i + batch_size]
        Y_batch = Y_train[i:i + batch_size]

        # Forward pass
        logits = net.forward(X_batch)

        # Compute loss
        sce = SigmoidCrossEntropy()
        loss = sce.forward(logits, Y_batch)
        losses.append(loss)

        # Backward pass
        grad_logits = sce.backward()
        net.backward(grad_logits)

        # Update weights
        net.step(step_size, momentum, weight_decay)

        # Compute accuracy
        predictions = np.where(logits < 0, [0], [1])
        acc = np.mean(predictions == Y_batch)
        accs.append(acc)

    t_loss, t_acc = evaluate(net, X_test_normalized, Y_test, batch_size)
    val_losses.append(t_loss)
    val_accs.append(t_acc)
    print(
        f'Epoch {epoch}, Loss: {np.mean(losses)}, Misclassification rate: {1 - np.mean(accs)},Test Loss: {t_loss}, Test Misclassification rate: {1 - t_acc}')

    # Plot training and testing curves
fig, ax1 = plt.subplots(figsize=(16, 9))
color = 'tab:red'
ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))], val_losses, c="red",
         label="Val. Loss")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_ylim(-0.01,3)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs, c="blue",
         label="Val. Acc.")
ax2.set_ylabel(" Accuracy", c=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(-0.01, 1.01)

plt.title('Best Model')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend(loc="center")
ax2.legend(loc="center right")
plt.savefig('./tuning_results/best_model.png')

print('Validation Accuracy : ' + str(max(val_accs)))

"""**Discussion:**
The model's accuracy on the training dataset kept decreasing where as the accuracy on the test dataset remained the same after some epochs. The best testing accuracy was 89.9% where as the best training accuracy for a lot of batches reached 1.0. This can be observed from the above plot.
"""
