import math as m
import numpy as np
import pandas as pd
 
class kohonens_network:
 
#computting which neuron has the best (the smallest) value
    def winning_neuron(self, weights, probe):
        neuron0 = 0
        neuron1 = 0
        for i in range(len(probe)):
            neuron0 = neuron0 + m.sqrt(m.pow((probe[i] - weights[0][i]), 2))
            neuron1 = neuron1 + m.sqrt(m.pow((probe[i] - weights[1][i]), 2))
            if neuron0 > neuron1:
                return 0
            else:
                return 1
#method updating which vector is a winner
    def update(self, weights, probe, champion_neuron, learning_rate):
        for i in range(len(weights)):
            weights[champion_neuron][i] = weights[champion_neuron][i] + learning_rate * (probe[i] - weights[champion_neuron][i])
        return weights
 
#executing the script 
def main():
 
#defining the data, by which the model will be trained
    training_data = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]]
	 
    m, n = len(training_data), len(training_data[0])
#computing random weights
    weights= []
    weights= list()
    for i in range(2):
        weights.append([])
        for j in range(m-1):
            random_weight= np.random.random()
            weights[i].append(random_weight)
 
#training the model
    model = kohonens_network() 
    iterations = 10
    learning_rate = 0.1
 
    for i in range(iterations):
        for j in range(m):
#taking a training probe
            probe = training_data[j]
#computing the winner
            champion_neuron = model.winning_neuron(weights, probe) 
#updating the winner
            weights = model.update(weights, probe, champion_neuron, learning_rate)
 
#using the model- checking the classification for test data
    while True:
        data_choice= input("Choose one option: \n Option num. 1: Run script with implemented test data. \n Option num. 2: Run script with user input data. \n I choose option num.: ")
        if data_choice== "1":
            test_probe= [1, 1, 1, 1]
            break
        elif data_choice== "2":
            test_probe = []
            n= int(input("Give a number of elements of an array: "))
            for _ in range(0, n):
                element_of_array=float(input("Write down an array's element: "))
                test_probe.append(element_of_array)
            break
        else:
            print("Something went wrong, try again.")

    champion_neuron = model.winning_neuron(weights, test_probe)
#presenting results of the script 
    print("\n Chosen weights: \n", pd.DataFrame(weights))
    print(f"\n Test probe {test_probe} is classified to the claster: ", champion_neuron)
 
#execution of the script 
if __name__ == "__main__":
    main()
