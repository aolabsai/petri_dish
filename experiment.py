import ao_core as ao

import os
os.chdir("AO/projects/petri_dish")
from env import PetriDish, Assay


env_size = 50
env_input_layers = 3

# Create a Petri dish instance

stimuli_dist = [
    {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0, 'max_p': 1},
    {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0, 'max_p': 1},
    {'type': 'linear', 'direction': 'horizontal-leftright', 'min_p': 0, 'max_p': 1}
]
dish = PetriDish(size=env_size, num_layers=env_input_layers, distributions=stimuli_dist)
dish.visualize()

input_channels = env_input_layers
input_channel_size = 9 # of binary neurons
arch = ao.Arch(
    arch_i=[input_channel_size]*input_channels, 
    arch_z=[1], 
    arch_c=[1], 
    connector_function="full_conn", 
    description="basic_clam")

def c0_instinct_rule(INPUT, Agent):
    
    input_pleasure_threshold = 3 # this number should be close to assay.sensory_binary_neurons and assay.sensory_radius

    if sum(INPUT[0:input_channel_size]) >= input_pleasure_threshold:
        instinct_response = [1, "c0 pleasure instinct"]
    else:
        instinct_response = [0, "c0 pass"]    
    return instinct_response            
arch.datamatrix[4, arch.C[1][0]] = c0_instinct_rule # Saving the function to the Arch so the Agent can access it

assay = Assay(petri_dish=dish, num_agents=10, start_logic='random', agent_archs=arch)
# load an Assay with agents from a previous Assay (moving worms from one petri dish to another)
# assay = Assay(petri_dish=dish, num_agents=10, start_logic='random', agent_archs=arch, assay_loadagents=assay)

assay.INSTINCTS = False
assay.set_agent_hyperparameters(
        C_impression_initial = 3, # strength of impression when first added to C_info
        C_impression_match = 1, # increment of impression in C_info if accessed
        C_pruning = 1, # decrement of impression in C_info if not accessed
        C_pruning_cutoff = 1, # value below which impressions are pruned from C_info
)

# # if circle sensory radius around agent
# assay.sensory_binary_neurons = 13
# assay.sensory_radius = 2

# if square
assay.sensory_binary_neurons = 9
assay.sensory_radius = 1 # this pair of numbers identical to conway's game of life

print("Visualizing the initial state of the assay...")
assay.visualize()

assay.random = False
assay.INSTINCTS = True # to activate training, let's gooooo
num_steps = 50 # run the simulation for 50 steps
print(f"\nRunning the assay for {num_steps} steps...")
for step in range(num_steps):
    assay.run_step()
print("...Done.")

# 4. Visualize the final state and agent paths
print("\nVisualizing the final state with agent paths...")
assay.visualize(show_paths=True)

dish.get_stimuli((49,49), radius=1, shape='square', mode='count')

all_data = assay.export_data() # resultant .pkl pickle file to be uploaded to the streamlit dashboard.py