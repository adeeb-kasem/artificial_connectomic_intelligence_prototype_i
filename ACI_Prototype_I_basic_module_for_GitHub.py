import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


#Step 1: Extract the connectivity data from the spreadsheet and convert it into a pandas DataFrame

# Read the spreadsheet into a DataFrame
spreadsheet_path = #Replace with filepath for CElegansConnectomeForModule.xlsx 
df = pd.read_excel(spreadsheet_path)

# Extract the relevant columns
connectivity_data = df[['pre_synaptic', 'post_synaptic', 'weight']]

# Make sure we have a count for all neurons by filling any NaN values with 0
connectivity_data = connectivity_data.fillna(0)

# build a dictionary to store the neuron_id, input_count, and output_count
neuron_dict = defaultdict(lambda: [0, 0])  # input_count, output_count

# perform a single pass over the connectivity_data DataFrame
for idx, row in connectivity_data.iterrows():
    neuron_dict[row['pre_synaptic']][1] += 1  # increment output count
    neuron_dict[row['post_synaptic']][0] += 1  # increment input count

# Print the extracted connectivity data
print(connectivity_data)


#Step 2: Create a list containing all unique neuron IDs

neuron_ids = pd.unique(connectivity_data[['pre_synaptic', 'post_synaptic']].values.ravel()).tolist()

# Print the list of unique neuron IDs
print(neuron_ids)


#Step 3: Initialize an empty matrix with dimensions N x N, where N is the total number of unique neuron IDs, and ensure that each row and column represents a specific neuron.

matrix_size = len(neuron_ids)

# Initialize an empty matrix
connectome_matrix = np.zeros((matrix_size, matrix_size))

# Print the initialized matrix
print(connectome_matrix)


#Step 4: Populate the matrix

# Iterate through the connectivity data and populate the matrix
for _, row in connectivity_data.iterrows():
    pre_neuron = row['pre_synaptic']
    post_neuron = row['post_synaptic']
    weight = row['weight']

    pre_index = neuron_ids.index(pre_neuron)
    post_index = neuron_ids.index(post_neuron)

    connectome_matrix[pre_index, post_index] = weight

# Print the populated matrix
print(connectome_matrix)


#Step 5: Distinguish between input neurons (sensory neurons in the original connectome) and output neurons (muscles in the original connectome).

# Create lists for input neuron IDs and output neuron IDs; fill in manually based on available data.

#outputs, muscles
# Used to remove from Axon firing since muscles cannot fire.
muscles = ['MVU', 'MVL', 'MDL', 'MVR', 'MDR']

muscleList = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23', 'MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']

mLeft = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23']
mRight = ['MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']
# Used to accumulate muscle weighted values in body muscles 07-23 = worm locomotion
musDleft = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23']
musVleft = ['MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23']
musDright = ['MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23']
musVright = ['MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']


#inputs, sensory neurons

somatosensory_neurons = ['FLPR', 'FLPL', 'ASHL', 'ASHR', 'IL1VL', 'IL1VR', 'OLQDL', 'OLQDR', 'OLQVR', 'OLQVL']

chemosensory_neurons = ['ADFL', 'ADFR', 'ASGR', 'ASGL', 'ASIL', 'ASIR', 'ASJR', 'ASJL']


#interneurons, neither sensory neurons nor muscles

interneurons = [neuron for neuron in neuron_ids if neuron not in muscleList and neuron not in somatosensory_neurons and neuron not in chemosensory_neurons]

# Print the list of interneurons
print("Interneurons:", interneurons)


#Step 6: Define a Neuron class that represents an individual neuron in your network.

class Neuron(nn.Module):
    def __init__(self, neuron_id, input_size, output_size):
        super(Neuron, self).__init__()
        self.neuron_id = neuron_id
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.fc(x)
        return out

# Create a dictionary to store all neurons, with neuron ID as the key
neurons = {}
for neuron_id, (input_count, output_count) in neuron_dict.items():
    neurons[neuron_id] = Neuron(neuron_id, input_count, output_count)

# Update your Neuron creation code to set input_size and output_size
for neuron_id in somatosensory_neurons:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron

for neuron_id in chemosensory_neurons:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron

for neuron_id in mLeft:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron
    
for neuron_id in mRight:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron

for neuron_id in musDleft:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron

for neuron_id in musVleft:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron

for neuron_id in musDright:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron

for neuron_id in musVright:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron

for neuron_id in interneurons:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1]  # Get the output_count from neuron_dict
    neuron = Neuron(neuron_id, input_size, output_size)
    neurons[neuron_id] = neuron


all_neurons = somatosensory_neurons + chemosensory_neurons + muscles + interneurons

for neuron_id in all_neurons:
    input_size = neuron_dict[neuron_id][0]  # Get the input_count from neuron_dict
    output_size = neuron_dict[neuron_id][1] 


#Step 7: Create a "Network" class

class Network(nn.Module):
    def __init__(self, neurons, activation_function = nn.ReLU()):
        super(Network, self).__init__()
        self.neurons = neurons
        self.activation_function = activation_function
        self.layers = self.create_layers()

    def create_layers(self):
        layers = []
        for neuron in self.neurons:
            layers.append(neuron)
        return nn.Sequential(*layers)

    def forward(self, input):
        output = self.layers(input)
        return self.activation_function(output)

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.layers = self.create_layers()

    def remove_neuron(self, neuron_id):
        self.neurons = [neuron for neuron in self.neurons if neuron.neuron_id != neuron_id]
        self.layers = self.create_layers()

    def set_activation_function(self, new_activation_function):
        self.activation_function = new_activation_function


print("Total neurons:", len(neurons))
print("Input neurons (Somatosensory):", len(somatosensory_neurons))
print("Input neurons (Chemosensory):", len(chemosensory_neurons))
print("Output neurons (Left muscles):", len(mLeft))
print("Output neurons (Right muscles):", len(mRight))
print("Output neurons (Dorsoventral muscles on the left side):", len(musDleft))
print("Output neurons (Ventral muscles on the left side):", len(musVleft))
print("Output neurons (Dorsoventral muscles on the right side):", len(musDright))
print("Output neurons (Ventral muscles on the right side):", len(musVright))
print("Interneurons:", len(interneurons))
print (neurons)