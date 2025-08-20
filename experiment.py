import ao_core as ao

import os
os.chdir("AO/projects/petri_dish")
from env import PetriDish, Assay


## Follow this script to set up an experiment. 
# 
# An experiment consists of 1) an environment, our simulated petri dish (a multi-layer 2D grid world, each layer representing different concentrations of stimuli), and 2) an assay or experimental setup (number of agents, their starting positions, etc.). Experiments should combine multiple environments and assays with the same agents.


## 1) Create Environment

# size of env (a square)
env_size = 100
env_input_layers = 3

# set distribution of stimuli in environment
stimuli_dist = [
    {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0, 'max_p': 1},
    {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0, 'max_p': 1},
    {'type': 'linear', 'direction': 'horizontal-leftright', 'min_p': 0, 'max_p': 1}
]

# create petri dish object, to be used as a param in assay below
dish = PetriDish(size=env_size, num_layers=env_input_layers, distributions=stimuli_dist)
dish.visualize()


## 2) Create Assay, or test over environment

input_channels = env_input_layers
input_channel_size = 9 # of binary neurons for each stimuli layer

# create Agent architecture (neuronal configuration)
arch = ao.Arch(
    arch_i=[input_channel_size]*input_channels, # 27 total input neurons
    arch_z=[1], # 1 output neuron, corresponding to: 1="go forward", 0="turn"
    arch_c=[1], # 1 custom control neuron, as instinct neuron for pleasure-from-food, defined below as c0
    connector_function="full_conn", # connection of neurons to each other
    description="basic_worm")

# create Agent's custom control function, corresponding to pleasure-from-food instinct
def c0_instinct_rule(INPUT, Agent):
    
    input_pleasure_threshold = 3 # this number should be close to assay.sensory_binary_neurons and assay.sensory_radius

    # instinct label
    instinct_label = 1 # when this instinct is triggered, it forces the Z neuron to be in this state
    z_nid = Agent.arch.Z__flat[0] # id of z neuron
    instinct_meta = "instinct"

    if sum(INPUT[0:input_channel_size]) >= input_pleasure_threshold:
        instinct_response = [1, "c0 pleasure instinct", [instinct_label, z_nid, instinct_meta]]
    else:
        instinct_response = [0, "c0 pass"]    
    return instinct_response            
arch.datamatrix[4, arch.C[1][0]] = c0_instinct_rule # saving the function to the Arch so the Agent can access it

# create assay object, to run tests from it
assay = Assay(petri_dish=dish, num_agents=10, start_logic='center', agent_archs=arch)
# OR you can also create an Assay that uses Agents from a previous Assay (like moving worms from one petri dish to another)
# assay = Assay(petri_dish=dish, num_agents=10, start_logic='center', agent_archs=arch, assay_loadagents=assay)

##### Experimental - hyperparameters related to forgetting, or the pruning of neuron-level memories over time

# only makes a difference if run with specific branch ao_core:research_expansion/neuron_pruning branch
assay.set_agent_hyperparameters(
        C_impression_initial = 5, # strength of impression when first added to neuron from C learning event
        C_impression_match = 2, # increment of impression if accessed by neuron during inference
        C_pruning = 2, # decrement of impression in C_info if not accessed by neuron during inference
        C_pruning_cutoff = 3, # value below which impressions are pruned from neuron
)

# Set sensory range of agent (how far agents can "see" around them) - 2 options available, circle or square
## if square
assay.sensory_shape = "square"
assay.sensory_binary_neurons = 9
assay.sensory_radius = 1 # this pair of numbers identical to conway's game of life
## if circle
# assay.sensory_shape = "circle"
# assay.sensory_binary_neurons = 13
# assay.sensory_radius = 2

print("Visualizing the initial state of the assay...")
assay.visualize()

# Run the assay simulation
assay.INSTINCTS = True # to activate training, let's gooooo
num_steps = 50
print(f"\nRunning the assay for {num_steps} steps...")
for step in range(num_steps):
    assay.run_step()
print("...Done.")

# Visualize the final state and agent paths
print("\nVisualizing the final state with agent paths...")
assay.visualize(show_paths=True)

# export data-- upload the resultant .pkl file to dashboard.py to view the data in a helpful dashboard
all_data = assay.export_data()


# View particular agent
num_agent = 14
agent = assay.agents[num_agent]["agent"]
# View particular neuron of agent
neuron = agent.neurons[agent.arch.Z__flat[0]]