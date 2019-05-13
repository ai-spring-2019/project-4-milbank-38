#______________________________________________________________________________
#
#   Project 4: Neural Network
#   Author: Oliver Keh
#   Date: 5/13/2019
#
#   There are 2 ways to run this program:
#   1. If you want to run the network once on a specific structure:
#      --> $ python3 project4.py --f <f> --n <n> --l <n> --v <v>
#   2. If you want to collect varied data using different sized networks:
#      --> $ python3 data_testing.py <f>
#
#   Where:                                              Requirement:
#   <f> = file name of desired dataset (str)            --> required
#   <n> = number of nodes per hidden layer (int)        --> optional
#   <l> = number of hidden layers (int)                 --> optional
#   <v> = use/don't use k-fold cross-validation (bool)  --> optional
#
#______________________________________________________________________________

import os
import csv, sys, random, math
import argparse

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs, full=False, multi=False):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. """

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)

        # Determine what subset of the results vector to use
        if full:
            actual = y
        else:
            actual = y[0]

        class_prediction = nn.predict_class(full, multi)
        if class_prediction != actual:
            true_positives += 1

    return 1 - (true_positives / total)

def component_sub(y, hw):
    vec = []

    for i in range(len(y)):
        diff = y[i] - hw[i]
        vec.append(diff)

    return vec

#______________________________________________________________________________
#
#   Neural Network classes
#
#______________________________________________________________________________

def learning_rate(epochs):
    """ A simple decaying learning rate based on the number of epochs"""
    return 1000 / (1000 + epochs)

class Node:

    def __init__(self):
        self.activation = 1.0

    def __str__(self):
        return str(self.activation)

    def __repr__(self):
        return str(self.activation)

    def set_input(self, input):
        self.activation = input

class DummyNode(Node):
    """ A subclass of the Node class that helps with differentiating
        between regular nodes and dummy nodes when output. """

    def __repr__(self):
        return "Dummy: " + str(self.activation)

    def __str__(self):
        return "Dummy: " + str(self.activation)

class NeuralNetwork:
    
    def __init__(self, layers):
        self.layers = self.create_network(layers)
        self.weights = self.create_weights(layers)

        self.inputs = self.layers[0]
        self.outputs = self.layers[-1]

    def set_inputs(self, values):

        wo_dummy = values[1:]

        for val in range(len(wo_dummy)):
            node = self.inputs[val]
            node.set_input(wo_dummy[val])

    def predict_class(self, full_result=False, multi_class=False):
        
        # With single-output network, make binary choice
        if len(self.outputs[:-1]) == 1:
            if self.outputs[0].activation <= 0.5:
                return 0
            return 1

        # Check if the entire (rounded) output layer is needed
        if full_result:

            result = []

            # Round outputs activations from all output nodes
            for node in self.outputs[:-1]:
                if node.activation <= 0.5:
                    result.append(0)
                else:
                    result.append(1)

            return result

        elif multi_class:
            closest_class = round(self.outputs[0].activation)
            return closest_class

        # Keep track of output node with highest probability
        max_activation = -1000000
        max_index = 0

        for i in range(len(self.outputs) - 1):
            if self.outputs[i].activation > max_activation:
                max_index = i 
                max_activation = self.outputs[i].activation

        return max_index

    def activation(self, value, log=False):
        
        if log:
            return logistic(value)
        else:
            if value >= 0:
                return 1
            else:
                return 0

    def calculate_inputs(self, node, layer):

        # Get the correct weight matrix
        weight_layer = self.weights[layer - 1]

        # Find inputs to specific node
        node_weights = weight_layer[node]

        # Find nodes in previous layer
        prev_nodes = self.layers[layer - 1]

        # Create a list of activations from nodes in previous layer
        activations = [ node.activation for node in prev_nodes ]

        # Return input to node i as the product of (w * x)
        return dot_product(node_weights, activations)

    def forward_propagate(self, input):

        # Set activations at input layer        
        self.set_inputs(input)

        # Iterate through layers except input layer
        for layer in range(1, len(self.layers)):

            # Get all nodes in current layer
            nodes = self.layers[layer]

            # Index through nodes in layer
            for i in range(len(nodes) - 1):

                # Calculate input at the ith node in the layer
                input = self.calculate_inputs(i, layer)
                
                # Set activation at node i
                nodes[i].activation = self.activation(input, True)
            
    def back_propagation_learning(self, training):

        epochs = 1

        while epochs < 1000:

            for pair in training:

                # Propagate the inputs forward to compute the outputs
                self.forward_propagate(pair[0])

                # Get activations of nodes in the output layer
                outputs = [ node.activation for node in self.outputs ]

                # Vector-wise subtraction to get errors at components in output
                err_vector = component_sub(pair[1], outputs)

                # Vector maintaining deltas 
                output_deltas = []

                # Compute deltas for each output node
                for k in range(len(self.outputs) - 1):

                    # Find kth node in output layer
                    node_k = self.outputs[k]

                    # Compute derivative of activation function (from class)
                    act_deriv = node_k.activation * (1 - node_k.activation)

                    # Compute error at output node k
                    err_k = err_vector[k] * act_deriv

                    # Add kth node error to vector of deltas
                    output_deltas.append(err_k)

                # Maintain new list of deltas for every layer
                layer_deltas = [output_deltas]

                # Start iteration from first hidden node layer
                back_layers = len(self.layers) - 2

                # Propagate deltas backward from output layer to input layer
                for i in range(back_layers, -1, -1):

                    # Deltas in current layer
                    curr_deltas = []

                    # Find weights between current layer and previous layer
                    weights_at_layer = self.weights[i]

                    # Iterate through nodes in current layer and modify deltas
                    for j in range(len(self.layers[i]) - 1):
                        
                        # Nodes in current layer
                        nodes = self.layers[i]

                        # Compute derivative of activation function (from class)
                        act_deriv = nodes[j].activation * (1 - nodes[j].activation)

                        # Use deltas from previously calculated layer, at pos 0
                        delta_k = layer_deltas[0]

                        # Get weights between current layer for node j
                        weights = [ weight[j] for weight in weights_at_layer ]

                        act_err = 0
                        for err in range(len(delta_k)):
                            act_err += delta_k[err] * weights[err]

                        # Add ith error to delta vector for current layer
                        curr_deltas.append(act_deriv * act_err)

                    layer_deltas.insert(0, curr_deltas)


                # Determine learning rate for current iteration
                rate = learning_rate(epochs)

                # Iterate through layers in network
                for l in range(len(self.weights)):

                    # Find all weights from layer i to layer j
                    weights_between_layers = self.weights[l]

                    # Update all weights in current layer
                    for j in range(len(weights_between_layers)):

                        weights = weights_between_layers[j]

                        for i in range(len(weights)):

                            # Find activation for node i in layer l
                            act = self.layers[l][i].activation

                            # Find delta for node j at layer l
                            delta_j = layer_deltas[l + 1][j]

                            weights[i] = weights[i] + (rate * act * delta_j)

            epochs += 1

    def create_network(self, layers):

        # Initialize empty array of node layers
        network = []

        for nodes in layers:

            # Create an extra node to account for dummy variable
            num_nodes = nodes

            # Generate desired number of nodes at current hidden layer
            curr_layer = [ Node() for i in range(num_nodes) ]

            # Append a dummy node to the end of the layer
            curr_layer.append(DummyNode())

            # Add the layer to the network
            network.append(curr_layer)

        return network

    def create_weights(self, layers):

        # Contains weight matrices at each layer in network
        weights = []

        for i in range(len(layers) - 1):
            
            # Create a matrix w/ dimensions: num nodes in layer x num weights
            matrix_at_layer = []

            # Number of rows = Number nodes in arrival layer 
            nodes_in_layer = layers[i + 1]

            # Number of cols = Number of nodes in departing layer + 1 dummy
            num_weights = layers[i] + 1

            for j in range(nodes_in_layer):

                # Create row of randomized weights
                node_weights = [ random.random() for x in range(num_weights) ]

                matrix_at_layer.append(node_weights)

            weights.append(matrix_at_layer)

        return weights

#______________________________________________________________________________
#
#   Cross-Validation
#
#______________________________________________________________________________


def k_fold_cross_validation(structure, data, full_output=False, multi_class=False, k=5):

    # Shuffle dataset
    random.shuffle(data)

    # Find size of each subset
    size = math.ceil(len(data) / k)

    # Split data into k-sized chunks
    subsets = [ data[i:i + size] for i in range(0, len(data), size) ]
    
    # Sum of accuracy over k iterations of validation
    sum = 0

    # Every k-sized chunk in data will be used as test 
    for i in range(0, len(subsets)):

        # Remove test chunk from training data
        set = [ j for j in subsets if j != subsets[i] ]

        # Join individual chunks in training data
        training = [item for subset in set for item in subset]

        # Set test data to be the ith k-chunk of the data
        test = subsets[i]

        # Initialize the neural network 
        nn = NeuralNetwork(structure)

        # Learn weights from training data
        nn.back_propagation_learning(training)

        # Maintain running sum of accuracy on unseen test data
        sum += accuracy(nn, test, full_output, multi_class)

    # Return average accuracy over k rounds 
    return sum / len(subsets)

#______________________________________________________________________________
#
#   Main function
#
#______________________________________________________________________________

def find_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", 
                        default="",
                        help="Filename")
    parser.add_argument("--v",
                        type=bool, 
                        default=False,
                        help="Use k-fold crossover validation",
                        required=False)
    parser.add_argument("--l",
                        type=int, 
                        default=0,
                        help="Number of hidden layers",
                        required=False)
    parser.add_argument("--n", 
                        type=int, 
                        default=0,
                        help="Nodes per hidden layer",
                        required=False)
    args = parser.parse_args()
    return args

def run_net(file, nodes=0, layers=0, validation=False):

    header, data = read_data(file, ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    #for example in training:
    #    print(example)

    # Determine input and output structures from data
    inputs = len(pairs[0][0])
    outputs = len(pairs[0][-1])

    # Create input, hidden layer, and output structure for neural network
    structure = [inputs] + [nodes for i in range(layers)] + [outputs]

    # Specify measurement parameters, note: only True in specific cases, ie. incrementer
    full = False
    multi = False

    if validation:
        score = k_fold_cross_validation(structure, training, full)
    else:
        nn = NeuralNetwork(structure)
        nn.back_propagation_learning(training)

        score = accuracy(nn, training, full)
    
    return score

def main():

    # Find all arguments (optionally required to run)
    args = find_args()
    
    # Run neural network on given file and parameters
    result = run_net(args.f, args.n, args.l, args.v)
    print("Result: ", result)

if __name__ == "__main__":
    main()
