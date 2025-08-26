import ao_core as ao

env_input_layers = 3

input_channels = env_input_layers
input_channel_size = 9 # of grid points around agent counted as input

def updateArch(stimuli_intensity, pain_threshold, pleasure_threshold):

    neurons_per_input_channel = input_channel_size * stimuli_intensity 
    # create Agent architecture (neuronal configuration)
    arch = ao.Arch(
        arch_i=[neurons_per_input_channel]*input_channels, # 27 total input neurons
        arch_z=[1], # 1 output neuron, corresponding to: 1="go forward", 0="turn"
        arch_c=[2], # 2 custom control neurons, as instinct neuron for pleasure-from-food and pain-from-hunger
        connector_function="full_conn", # connection of neurons to each other
        description="basic_worm")

    arch.C_types_names = ["Pleasure", "Pain"]

    input_pain_threshold = neurons_per_input_channel * pain_threshold
    input_pleasure_threshold = neurons_per_input_channel * pleasure_threshold

    # create Agent's custom control function, corresponding to pleasure-from-food instinct
    def c0_instinct_rule(INPUT, Agent):
        
        # instinct label
        instinct_label = 1 # when this instinct is triggered, it forces the Z neuron to be in this state
        z_nid = Agent.arch.Z__flat[0] # id of z neuron
        instinct_meta = "instinct - pleasure"

        if sum(INPUT[0:neurons_per_input_channel]) >= input_pleasure_threshold:
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

        if sum(INPUT[0:neurons_per_input_channel]) <= input_pain_threshold:
            instinct_response = [1, "c1 pain instinct", [z_nid, instinct_label, instinct_meta]]
            Agent.Pain += 1
        else:
            instinct_response = [0, "c0 pass"]

        return instinct_response            
    arch.datamatrix[4, arch.C[1][1]] = c1_instinct_rule # saving the function to the Arch so the Agent can access it

    return arch