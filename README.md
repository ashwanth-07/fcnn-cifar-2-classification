# FCNN for CIFAR-2 Classification

This project contains a Python implementation of a Fully Connected Neural Network (FCNN) from scratch for CIFAR-2 classification. I present the results of experimenting with different hyperparameters, including learning rates, batch sizes, and the number of hidden layers. The models were trained for 100 epochs, and the testing accuracy was recorded and plotted for each epoch for all different values of the hyperparameters.

## Model Architecture

The deep learning model used in this assignment is a feed-forward neural network. The number of hidden layers varied as one of the hyperparameters. Each layer is a linear layer which uses the ReLU activation function except the last layer.

## Hyperparameters

Three key hyperparameters were varied in this assignment:
1. Learning Rate: The learning rates tested include [0.1, 0.01, 0.001, 0.0001].
2. Batch Size: The batch sizes tested include [8, 16, 32, 64, 128].
3. Number of Hidden Layers: The numbers of hidden layers tested include [1, 2, 3, 4, 5].

## Dataset

The CIFAR-2 dataset is a binary classification version of the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. In CIFAR-2, we only consider two classes for binary classification.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

## Usage

1. Clone the repository:
    ```
    https://github.com/ashwanth-07/fcnn-cifar-2-classification.git
    ```
2. Navigate to the cloned project:
    ```
    cd fcnn-cifar-2-classification
    ```
3. Run the main file:
    ```
    python fully-connected-neural-network.py
    ```

## Results

The learning rate, batch size, and the number of hidden layers had a significant impact on the model's performance. The testing accuracy for each epoch for different hyperparameters is shown in the figures below (please add the figures).

The best hyperparameters were picked and a model was trained with a learning rate of 0.01, 3 hidden layers, a batch size of 8, momentum equals 0.9 and weight decay is 0.000001. The training was done for 300 epochs and a training accuracy of 94% and testing accuracy of 83.5% were achieved.
