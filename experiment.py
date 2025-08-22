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
stimuli_intensity = 1


# set distribution of stimuli in environment
stimuli_distribution = [
    {'type': 'quadrant', 'quadrant': 'bottom-right', 'peak': 'corner', 'diffusion': 'linear', 'min_p': 0, 'max_p': 1},
    # {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0, 'max_p': 1},
    {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0, 'max_p': 1},
    {'type': 'linear', 'direction': 'horizontal-leftright', 'min_p': 0, 'max_p': 1}
]


# stimuli_distribution = [
#     # Layer 0: A complex layer with three active quadrants
#     {
#         'type': 'quadrant',
#         # 'start_pos': (0.25, 0.25), # Active in a central square
#         # 'end_pos': (0.75, 0.75),
#         'quadrant_setups': [
#             # Setup for the top-left quadrant
#             {
#                 'quadrant': 'top-left',
#                 'peak': 'corner',          # Highest concentration is at the corner (0,0)
#                 'diffusion': 'linear',     # Gradient decreases linearly towards the center
#                 'min_p': 0.1,              # Probability at the center boundary
#                 'max_p': 0.9               # Probability at the far corner
#             },
#             # Setup for the bottom-right quadrant
#             {
#                 'quadrant': 'bottom-right',
#                 'peak': 'center',          # Highest concentration is near the center (0.5, 0.5)
#                 'diffusion': 'radial',     # Gradient decreases radially towards the corner
#                 'min_p': 0.0,              # Probability at the corner boundary
#                 'max_p': 0.8               # Probability at the center
#             },
#             # Setup for the top-right quadrant
#             {
#                 'quadrant': 'top-right',
#                 'peak': 'center',
#                 'diffusion': 'linear',
#                 'min_p': 0.2,
#                 'max_p': 0.5
#             }
#             # The bottom-left quadrant is left undefined, so it will have zero stimuli.
#         ]
#     },
    
#     # Layer 1: A simple vertical linear gradient for comparison
#     # {
#     #     'type': 'linear', 
#     #     'direction': 'vertical-downup', 
#     #     'min_p': 0.0, 
#     #     'max_p': 0.5
#     #     'start_pos': (0.0, 0.0), # Define the active area's top-left corner
#     #     'end_pos': (0.5, 0.5)    # Define the active area's bottom-right corner
#     # },
#     {
#         'type': 'radial',
#         'position': 'top-left',  # Peak concentration at the top-left of the active area
#         'min_p': 0.0,
#         'max_p': 1.0,
#         # 'start_pos': (0.0, 0.0), # Define the active area's top-left corner
#         'end_pos': (0.5, 0.75)    # Define the active area's bottom-right corner
#     },
    
#     # Layer 2: A simple radial gradient for comparison
#     {
#         'type': 'linear',
#         'direction': 'horizontal-rightleft',
#         'min_p': 0.1,
#         'max_p': 0.8,
#         'start_pos': (0.25, 0.25), # Active in a central square
#         'end_pos': (0.75, 0.75)
#     }
# ]

# create petri dish object, to be used as a param in assay below
dish = PetriDish(size=env_size, num_layers=env_input_layers, distributions=stimuli_distribution, stimuli_intensity=stimuli_intensity)
dish.visualize()


## 2) Create Assay, or test over environment

input_channels = env_input_layers
input_channel_size = 9 # of grid points around agent counted as input
input_intensity = stimuli_intensity
input_channel_neurons = input_channel_size * input_intensity # for 1 layer

# create Agent architecture (neuronal configuration)
arch = ao.Arch(
    arch_i=[input_channel_neurons]*input_channels, # 27 total input neurons
    arch_z=[1], # 1 output neuron, corresponding to: 1="go forward", 0="turn"
    arch_c=[2], # 2 custom control neurons, as instinct neuron for pleasure-from-food and pain-from-hunger
    connector_function="full_conn", # connection of neurons to each other
    description="basic_worm")

arch.C_types_names = ["Pleasure", "Pain"]

input_pain_threshold = input_channel_neurons * 1/3
input_pleasure_threshold = input_channel_neurons * 2/3

# create Agent's custom control function, corresponding to pleasure-from-food instinct
def c0_instinct_rule(INPUT, Agent):
    
    # instinct label
    instinct_label = 1 # when this instinct is triggered, it forces the Z neuron to be in this state
    z_nid = Agent.arch.Z__flat[0] # id of z neuron
    instinct_meta = "instinct - pleasure"

    if sum(INPUT[0:input_channel_neurons]) >= input_pleasure_threshold:
        instinct_response = [1, "c0 pleasure instinct", [z_nid, instinct_label, instinct_meta]]
        Agent.Pleasure += 1
    else:
        instinct_response = [0, "c0 pass"]    
    
    return instinct_response            
arch.datamatrix[4, arch.C[1][0]] = c0_instinct_rule # saving the function to the Arch so the Agent can access it

# create Agent's custom control function, corresponding to pain-from-lack-of-food instinct
def c1_instinct_rule(INPUT, Agent):

    # instinct label
    instinct_label = 0 # when this instinct is triggered, it forces the Z neuron to be in this state
    z_nid = Agent.arch.Z__flat[0] # id of z neuron
    instinct_meta = "instinct - pain"

    if sum(INPUT[0:input_channel_neurons]) <= input_pain_threshold:
        instinct_response = [1, "c1 pain instinct", [z_nid, instinct_label, instinct_meta]]
        Agent.Pain += 1
    else:
        instinct_response = [0, "c0 pass"]

    return instinct_response            
arch.datamatrix[4, arch.C[1][1]] = c1_instinct_rule # saving the function to the Arch so the Agent can access it

# create assay object, to run tests from it
assay = Assay(petri_dish=dish, num_agents=100, start_logic='center', agent_archs=arch, save_agent_meta=False)
# OR you can also create an Assay that uses Agents from a previous Assay (like moving worms from one petri dish to another)
# previous_assay = assay
# assay = Assay(petri_dish=dish, num_agents=100, start_logic='center', agent_archs=arch, assay_loadagents=previous_assay)

##### Experimental - hyperparameters related to forgetting, or the pruning of neuron-level memories over time
# only makes a difference if run with specific branch ao_core:research_expansion/neuron_pruning branch
assay.set_agent_hyperparameters(
        C_impression_initial = 5, # strength of impression when first added to neuron from C learning event
        C_impression_match = 2, # increment of impression if accessed by neuron during inference
        C_pruning = 1, # decrement of impression in C_info if not accessed by neuron during inference
        C_pruning_cutoff = 1, # value below which impressions are pruned from neuron
)

# num_steps = 10
# assay.pretrain_random(num_steps)

# Run the assay simulation
assay.INSTINCTS = True # to activate training, let's gooooo
num_steps = 50
print(f"\nRunning the assay for {num_steps} steps...")
assay.run_step(num_steps)

# Visualize the final state and agent paths
print("\nVisualizing the final state with agent paths...")
assay.visualize(show_paths=True)

# export data-- upload the resultant .pkl file to dashboard.py to view the data in a helpful dashboard
all_data = assay.export_data()


# View particular agent / neuron
num_agent = 0 # if of agent in assay
agent = assay.agents[num_agent]["agent"]
# View particular neuron of agent
neuron = agent.neurons[agent.arch.Z__flat[0]]