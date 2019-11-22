import nn
import numpy as np
import time

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dotProduct = nn.as_scalar(self.run(x))
        if dotProduct < 0:
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converged = False
        while not converged:
            failures = 0
            for item, classification in dataset.iterate_once(1):
                prediction = self.get_prediction(item)
                if prediction != nn.as_scalar(classification):
                    failures += 1
                    self.w.update(item, nn.as_scalar(classification))
            if failures == 0:
                converged = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = -0.05

        self.layer1 = nn.Parameter(1, 100)
        self.bias1 = nn.Parameter(1, 100)

        self.layer2 = nn.Parameter(100, 100)
        self.bias2 = nn.Parameter(1, 100)

        self.layer3 = nn.Parameter(100, 1)
        self.bias3 = nn.Parameter(1, 1)

        self.parameters = [self.layer1, self.bias1, self.layer2, self.bias2, self.layer3, self.bias3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.layer1), self.bias1))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.layer2), self.bias2))
        layer3 = nn.AddBias(nn.Linear(layer2, self.layer3), self.bias3)
        return layer3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictedY = self.run(x)
        return nn.SquareLoss(predictedY, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        lossPercentage = 1.0
        while lossPercentage >= 0.02:
            for data, classifications in dataset.iterate_once(10):
                loss = self.get_loss(data, classifications)
                gradients = nn.gradients(loss, self.parameters)
                for index, parameter in enumerate(self.parameters):
                    parameter.update(gradients[index], self.learningRate)
            for data, classifications in dataset.iterate_once(200):
                lossPercentage = nn.as_scalar(self.get_loss(data, classifications))

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = -0.05

        self.layer1 = nn.Parameter(784, 784)
        self.bias1 = nn.Parameter(1, 784)

        self.layer2 = nn.Parameter(784, 1000)
        self.bias2 = nn.Parameter(1, 1000)

        self.layer3 = nn.Parameter(1000, 500)
        self.bias3 = nn.Parameter(1, 500)

        self.layer4 = nn.Parameter(500, 10)
        self.bias4 = nn.Parameter(1, 10)

        self.parameters = [self.layer1, self.bias1, self.layer2, self.bias2, self.layer3, self.bias3, self.layer4, self.bias4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.layer1), self.bias1))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.layer2), self.bias2))
        layer3 = nn.ReLU(nn.AddBias(nn.Linear(layer2, self.layer3), self.bias3))
        layer4 = nn.AddBias(nn.Linear(layer3, self.layer4), self.bias4)
        return layer4

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictedY = self.run(x)
        return nn.SoftmaxLoss(predictedY, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        lossPercentage = 1.0
        while lossPercentage >= 0.03:
            for data, classifications in dataset.iterate_once(25):
                loss = self.get_loss(data, classifications)
                gradients = nn.gradients(loss, self.parameters)
                for index, parameter in enumerate(self.parameters):
                    parameter.update(gradients[index], self.learningRate)
            lossPercentage = 1.0 - dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = 0.10
        #        previous output --> previous layer -----> + --> output layer --> output
        #            (b x 5)            (b x 50)           ^      (50 x 5)        (b x 5)
        #                                                  |
        #                                                  |
        #                                             input layer (47 x 50)
        #                                                  ^
        #                                                  |
        #                                                input  (b x 47)

        self.previousOutputLayers = [nn.Parameter(128, 128), nn.Parameter(128, 128)]
        self.previousOutputBiases = [nn.Parameter(1, 128), nn.Parameter(1, 128)]
        
        self.inputLayers = [nn.Parameter(47, 128), nn.Parameter(128, 128)]
        self.inputBiases = [nn.Parameter(1, 128) , nn.Parameter(1, 128)]

        self.outputLayers = [nn.Parameter(128, 128), nn.Parameter(128, 5)]
        self.outputBiases = [nn.Parameter(1, 128), nn.Parameter(1, 5)]

        self.parameters = []
        self.parameters.extend(self.previousOutputLayers)
        self.parameters.extend(self.previousOutputBiases)
        self.parameters.extend(self.inputLayers)
        self.parameters.extend(self.inputBiases)
        self.parameters.extend(self.outputLayers)
        self.parameters.extend(self.outputBiases)

    def runInputLayer(self, data):
        inputLayer = None
        for biasIndex, layer in enumerate(self.inputLayers):
            inputLayer = nn.ReLU(nn.AddBias(nn.Linear(data, layer), self.inputBiases[biasIndex]))
            data = inputLayer
        return inputLayer

    def runPreviousOutputLayer(self, data):
        previousOutputLayer = None
        for biasIndex, layer in enumerate(self.previousOutputLayers):
            previousOutputLayer = nn.ReLU(nn.AddBias(nn.Linear(data, layer), self.previousOutputBiases[biasIndex]))
            data = previousOutputLayer
        return previousOutputLayer

    def runOutputLayer(self, data):
        outputLayer = None
        for biasIndex, layer in enumerate(self.outputLayers[:-1]):
            outputLayer = nn.ReLU(nn.AddBias(nn.Linear(data, layer), self.outputBiases[biasIndex]))
            data = outputLayer
        outputLayer = nn.AddBias(nn.Linear(data, self.outputLayers[-1]), self.outputBiases[-1])
        return outputLayer

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        previousOutput = None
        for letter in xs:
            inputLayer = self.runInputLayer(letter)
            if previousOutput is not None:
                previousOutputLayer = self.runPreviousOutputLayer(previousOutput)
                inputLayer = nn.Add(inputLayer, previousOutputLayer)
            outputLayer = self.runOutputLayer(inputLayer)
            previousOutput = inputLayer
        return outputLayer

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictedY = self.run(xs)
        return nn.SoftmaxLoss(predictedY, y)
        # return nn.SquareLoss(predictedY, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        lossPercentage = 1.0
        epoch = 0
        start = time.time()
        while lossPercentage >= 0.05:
            if epoch == 20:
                print("Reached {0:.2f}% loss in {1} epochs. {2} seconds elapsed".format(lossPercentage, epoch, time.time() - start))
                break
            for data, classifications in dataset.iterate_once(50):
                loss = self.get_loss(data, classifications)
                gradients = nn.gradients(loss, self.parameters)
                for index, parameter in enumerate(self.parameters):
                    parameter.update(gradients[index], -self.learningRate)
            lossPercentage = 1.0 - dataset.get_validation_accuracy()
            print("Epoch {0} finished with {1:.2f}% accuracy. {2} seconds elapsed".format(epoch, 1.0 - lossPercentage, time.time() - start))
            epoch += 1
            
