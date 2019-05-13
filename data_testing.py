#______________________________________________________________________________
#
#   Project 4: Neural Network
#   Author: Oliver Keh
#   Date: 5/13/2019
#
#   This is a program that automates the data collection process. Using a 
#   standardized number of hidden layers and nodes, it creates a table of
#   data that is used in the report.
#______________________________________________________________________________

import sys, os, csv, datetime
import project4

def write_results(file, performance):
    results = zip(*performance.values())
    with open(file, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(performance.keys())
        writer.writerows(results)

def main():
    file = sys.argv[1]

    print("\nRunning neural network on: {}".format(file))
    print("Starting at: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    max_hidden_layers = 3
    hidden_nodes = [1, 2, 3, 4, 5, 6, 12]
    validation = True

    performance = {}

    # Find results for base case with zero hidden nodes/layers
    print("Parameters: layers={}, nodes={}, validation={}".format(0, 0, validation))
    basic_result = project4.run_net(file, 0, 0, validation)
    print("     Result={}%".format(basic_result))

    performance[0] = [round(basic_result * 100, 2)] + (['-'] * len(hidden_nodes))

    for layer in range(1, max_hidden_layers + 1):

        layer_results = ['-']

        for nodes in hidden_nodes:
            print("Parameters: layers={}, nodes={}, validation={}".format(layer, nodes, validation))
            # Run neural network with parameters
            result = project4.run_net(file, nodes, layer, validation)
            print("     Result={}%".format(round(result * 100, 2)))

            # Convert result to percentage with two decimal places
            percentage = round(result * 100, 2)
            layer_results.append(percentage)

        performance[layer] = layer_results

    # Determine new file name for results
    new_filename = "results-" + file
    print("Completed run. Outputting to: {}\n".format(new_filename))
    print("Finishing time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    # Write output to csv file
    write_results(new_filename, performance)

if __name__ == '__main__':
    main()