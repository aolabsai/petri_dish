import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from matplotlib.colors import ListedColormap

from archs.arch0 import updateArch

from main_classes import Assay

st.set_page_config(
    page_title="AO | Petri Dish Benchmark",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- STREAMLIT APP UI AND LOGIC ---

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload experiment data",
    type=["pkl"],
    help="Upload a pickle file containing agent experiment data."
)
st.title("Continuous Learning Benchmark: Petri Dish Simulation")

# --- Initialize Session State ---
if 'dish' not in st.session_state:
    st.session_state.dish = []

if not st.session_state.dish:
    st.info("⬅️ Go to the env page to set an env first.")
else:

    with st.expander("Run a New Simulation", expanded=True):
        st.markdown("""
        Configure and run a new simulation. The results will be displayed in the views below.
        """)
        debug_mode_checkbox = st.checkbox("Enable debug mode", value=False, help='If checked, agents will move randomly (sets agent_archs="random").')
        reuse_agents_from_assay = st.checkbox("Reuse agents from previous assay", value=False, help="If checked, assay will be run with agents from previous trial. WNN agent are natively stateful with memories are persistent across assays.", disabled="assay" not in st.session_state)
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            num_agents_input = st.slider("Number of Agents", min_value=1, max_value=100, value=10, help="Select the number of agents to include in the simulation.", disabled=reuse_agents_from_assay)
        with sim_col2:
            start_logic_input = st.selectbox("Agent Starting Positions", options=['random', 'center', 'cardinal', 'quadrants', 'corners'], index=0, help="Choose the initial placement pattern for the agents.")
        with sim_col3:
            steps_input = st.slider("Simulation Steps", min_value=1, max_value=1000, value=10, help="Set the total number of time steps for the simulation.")

    with st.expander("Set Agent-level hyperparameters:", expanded=False):
        st.markdown("""
        Neurons' lookup tables maintain a record of how often each row is used for inference during CGA (in ao.agent.neuron.c_info). This information is used to prune lookup tables so that stale, unused rows are deleted, the desired emergent (or agent-level affect) being the "forgetting" component of classical conditioning. Set those parameters below:
        """)
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
                C_impression_initial = st.number_input("Initial learning experience impression strength", min_value=1, max_value=100, value=5, help="The number added to the memory'") # strength of impression when first added to neuron from C learning event
                C_impression_match = st.number_input("Impression increment if lookup match", min_value=1, max_value=C_impression_initial, value=2, help="The number incremented when there is a lookup match")  # increment of impression if accessed by neuron during inference
                C_pruning = st.number_input("Impression decrement for all other rows that did not match", min_value=1, max_value=C_impression_match, value=1, help="The number decremented when no lookup match") # decrement of impression in C_info if not accessed by neuron during inference
                C_pruning_cutoff = st.number_input("Number below which memories are deleted from neurons' lookup tables", min_value=1, max_value=100, value=1, help="Cutoff value below which memories are deleted") # value below which impressions are pruned from neuron)
                

    if st.button("▶️ Run Simulation"):
        with st.spinner(f"Running simulation for {steps_input} steps..."):
            petri_dish = st.session_state.dish
            agent_archs_param = "random" if debug_mode_checkbox else updateArch(st.session_state.stimuli_intensity) # the arch that is imported from the "archs" folder
            
            if reuse_agents_from_assay and "assay" in st.session_state:
                assay_loadagents = st.session_state.assay
            else:
                assay_loadagents = ""
            assay = Assay(petri_dish=petri_dish, num_agents=num_agents_input, start_logic=start_logic_input, agent_archs=agent_archs_param, steps=steps_input, assay_loadagents=assay_loadagents)
            assay.set_agent_hyperparameters(
                C_impression_initial, # strength of impression when first added to neuron from C learning event
                C_impression_match, # increment of impression if accessed by neuron during inference
                C_pruning, # decrement of impression in C_info if not accessed by neuron during inference
                C_pruning_cutoff, # value below which impressions are pruned from neuron)
)
            assay.INSTINCTS = True # to activate training, let's gooooo
            assay.run_step(steps=steps_input)
            simulation_data = assay.export_data()
            st.session_state.simulation_data = simulation_data
            st.session_state.assay = assay
            st.success("Simulation complete! Results are displayed below.")


    # --- Visualization Helper Functions ---

    def visualize_layer_data(layer, layer_index, num_layers, max_intensity):
        """Generates an RGB numpy array for a single layer with a white-to-color gradient."""
        color = colorsys.hsv_to_rgb(layer_index / num_layers, 1, 1)
        
        # Create a colormap from white to the layer's primary color
        cmap_colors = np.array([np.linspace(1, c, int(max_intensity) + 1) for c in color]).T
        layer_cmap = ListedColormap(cmap_colors)

        # Normalize the layer data and apply the colormap
        norm_layer = layer / max_intensity
        rgb_data = layer_cmap(norm_layer)

        # Return as an 8-bit RGB array
        return (rgb_data[:, :, :3] * 255).astype(np.uint8)

    def visualize_combined_data(layers, max_intensity):
        """Generates a combined RGB numpy array for all layers using an additive color model."""
        if not layers:
            return np.zeros((1, 1, 3), dtype=np.uint8)
            
        num_layers = len(layers)
        grid_size = layers[0].shape[0]
        colors = [colorsys.hsv_to_rgb(i / num_layers, 1, 1) for i in range(num_layers)]
        
        # Start with a black canvas for additive color blending
        combined_rgb = np.zeros((grid_size, grid_size, 3), dtype=float)
        
        # Handle case where max_intensity is 0 to avoid division by zero
        normalized_max = float(max_intensity) if max_intensity > 0 else 1.0

        for i, layer in enumerate(layers):
            # Normalize intensity to create a brightness map for this layer's color
            normalized_intensity = layer.astype(float) / normalized_max
            color_array = np.array(colors[i])
            
            # Add this layer's color, scaled by its intensity, to the combined image
            combined_rgb += normalized_intensity[:, :, np.newaxis] * color_array

        # Clip values to the valid RGB range [0, 1] and convert to 8-bit
        combined_rgb = np.clip(combined_rgb, 0, 1)
        return (combined_rgb * 255).astype(np.uint8)

    def plot_grid(layer_rgb, combined_data, agent_colors, selected_agents, grid_size):
        """Plots the environment background and agent paths on top."""
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor((51/255, 51/255, 51/255))
        ax.imshow(layer_rgb, origin='upper') # Fix: Use 'upper' to match standard image orientation
        for agent in selected_agents:
            agent_data = combined_data[combined_data['Agent'] == agent]
            if not agent_data.empty:
                # Note: Plotting pos_y vs pos_x to align with imshow's (row, col) convention
                ax.plot(agent_data['pos_y'], agent_data['pos_x'], color=agent_colors[agent], linewidth=1.5, alpha=0.8)
                start_pos = agent_data.iloc[0]
                ax.scatter(start_pos['pos_y'], start_pos['pos_x'], marker='o', s=20, color=agent_colors[agent], edgecolor='white', linewidth=0.5, zorder=5)
                end_pos = agent_data.iloc[-1]
                ax.scatter(end_pos['pos_y'], end_pos['pos_x'], marker='X', s=60, color=agent_colors[agent], edgecolor='white', linewidth=1, zorder=5)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)     # Adjust ylim to match the inverted y-axis of 'imshow' with origin='upper'
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0)
        return fig


    # --- Main Data Processing and Visualization Logic ---

    data_source = None
    if 'simulation_data' in st.session_state:
        data_source = st.session_state.simulation_data
    elif uploaded_file is not None:
        data_source = pd.read_pickle(uploaded_file)

    if data_source is not None:
        try:
            all_data = data_source
            agent_info, env_data = all_data[0], all_data[1]
            agent_names, agent_dfs = agent_info[0], agent_info[1]
            
            full_data = pd.concat([df.assign(Agent=name) for df, name in zip(agent_dfs, agent_names)], ignore_index=True)
            grid_size = env_data[0].shape[0]

            required_cols = {'Time', 'Experience States', 'pos_x', 'pos_y', 'Agent',
                            'stimuli0', 'stimuli1', 'stimuli2',
                            'neuronal0', 'neuronal1', 'neuronal2'}
            
            learning_event_cols = [col for col in full_data.columns if col in ['Pleasure', 'Pain']]
                
            if not required_cols.issubset(full_data.columns):
                missing_cols = required_cols - set(full_data.columns)
                st.error(f"Data Error: The DataFrame is missing required columns: {', '.join(missing_cols)}")
                st.stop()

        except Exception as e:
            st.error(f"An error occurred while reading or processing the data: {e}")
            st.stop()

        agents = agent_names
        st.sidebar.header("Agent Selection")
        selected_all_agents = st.sidebar.checkbox("Display all Agents", value=True)
        if selected_all_agents:
            selected_agents = agents
        else:
            selected_agents = st.sidebar.multiselect("Or select Agents to display", agents, default=agents)

        if selected_agents:
            combined_data = full_data[full_data['Agent'].isin(selected_agents)].copy()

            color_tuples = plt.cm.tab10.colors
            agent_colors = {
                agent: mcolors.to_hex(color_tuples[i % len(color_tuples)]) 
                for i, agent in enumerate(agents)
            }
            
            try:
                layers = env_data
                num_layers = len(layers)
            except (IndexError, TypeError) as e:
                st.error(f"Could not read stimuli layers from the data. Error: {e}")
                st.stop()

            # Generate RGB data for each layer and the combined view
            max_intensity = float(max(layer.max() for layer in layers if layer.size > 0) or 1)
            layer1_rgb = visualize_layer_data(layers[0], 0, num_layers, max_intensity)
            layer2_rgb = visualize_layer_data(layers[1], 1, num_layers, max_intensity)
            layer3_rgb = visualize_layer_data(layers[2], 2, num_layers, max_intensity)
            layer4_rgb = visualize_combined_data(layers, max_intensity)

            st.markdown("---")

            _left_spacer, center_col, _right_spacer = st.columns([1, 1, 1])
            with center_col:
                st.header("Agents Movement Over Time")
                legend_text = " **Legend:** " + " | ".join([f"<span style='color:{agent_colors[agent]}; font-weight:bold;'>⦿ {agent}</span>" for agent in selected_agents])
                st.markdown(legend_text, unsafe_allow_html=True)
        
                with st.expander("Environment View (Combined)", expanded=True):
                    color_scale = alt.Scale(domain=list(agent_colors.keys()), range=list(agent_colors.values()))
                    st.pyplot(plot_grid(layer4_rgb, combined_data, agent_colors, selected_agents, grid_size))
            with st.expander("Env Views by Stimuli (Filtered)", expanded=False):
                grid_col1, grid_col2, grid_col3 = st.columns(3)
                # Assign colors based on layer index, matching the PetriDish class logic
                layer_titles = [f"Layer {i+1}" for i in range(num_layers)]
                layer_rgbs = [layer1_rgb, layer2_rgb, layer3_rgb]
                
                for i, col in enumerate([grid_col1, grid_col2, grid_col3]):
                    with col:
                        st.subheader(layer_titles[i])
                        st.pyplot(plot_grid(layer_rgbs[i], combined_data, agent_colors, selected_agents, grid_size))

            st.header("Agent Activity Over Time")
            with st.expander("I (input) Neuron Activations", expanded=True):
                s_col1, s_col2, s_col3 = st.columns(3)
                layer_titles = ["Layer 1: Red", "Layer 2: Green", "Layer 3: Blue"]
                stimuli_cols = ['stimuli0', 'stimuli1', 'stimuli2']
                for i, col in enumerate([s_col1, s_col2, s_col3]):
                    with col:
                        st.subheader(layer_titles[i])
                        response_chart = alt.Chart(combined_data).mark_line().encode(
                            x=alt.X('Time', axis=alt.Axis(title=None)), y=alt.Y(stimuli_cols[i], axis=alt.Axis(title=None)),
                            color=alt.Color('Agent', scale=color_scale, legend=None), tooltip=['Time', stimuli_cols[i], 'Agent']
                        ).interactive()
                        st.altair_chart(response_chart, use_container_width=True)

            with st.expander("Q (inner) Neuron Activations", expanded=True):
                n_col1, n_col2, n_col3 = st.columns(3)
                neuronal_cols = ['neuronal0', 'neuronal1', 'neuronal2']
                for i, col in enumerate([n_col1, n_col2, n_col3]):
                    with col:
                        st.subheader(layer_titles[i])
                        neuronal_chart = alt.Chart(combined_data).mark_line().encode(
                            x=alt.X('Time', axis=alt.Axis(title=None)), y=alt.Y(neuronal_cols[i], axis=alt.Axis(title=None)),
                            color=alt.Color('Agent', scale=color_scale, legend=None), tooltip=['Time', neuronal_cols[i], 'Agent']
                        ).interactive()
                        st.altair_chart(neuronal_chart, use_container_width=True)
            
            st.markdown("---")
            with st.expander("Learning Events and Memory", expanded=True):
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("Control Events")
                    st.text("Total number of learning events over time.")
                    if set(learning_event_cols) == {'Pleasure', 'Pain'}:
                        if st.checkbox("View combined plot (Pleasure - Pain)"):
                            combined_data['Combined_Events'] = combined_data['Pleasure'] - combined_data['Pain']
                            combined_chart = alt.Chart(combined_data).mark_line().encode(
                                x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Combined_Events', axis=alt.Axis(title="Net Events")),
                                color=alt.Color('Agent', scale=color_scale, legend=None), tooltip=['Time', 'Agent', 'Combined_Events']
                            ).interactive()
                            st.altair_chart(combined_chart, use_container_width=True)
                            st.text("Displaying the net of Pleasure minus Pain.")
                        else:
                            selected_event_types = learning_event_cols
                            learning_data_long = combined_data.melt(id_vars=['Time', 'Agent'], value_vars=selected_event_types, var_name='Event Type', value_name='Count')
                            learning_chart = alt.Chart(learning_data_long).mark_line(interpolate='step-after').encode(
                                x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Count', axis=alt.Axis(title="Total Events")),
                                color=alt.Color('Agent', scale=color_scale, legend=None), strokeDash=alt.StrokeDash('Event Type', legend=alt.Legend(title="Event Type")),
                                tooltip=['Time', 'Agent', 'Event Type', 'Count']
                            ).interactive()
                            st.altair_chart(learning_chart, use_container_width=True)

                with col4:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.subheader("Experience States")
                    st.text("Total unique memories in the output neuron.")
                    states_chart = alt.Chart(combined_data).mark_line().encode(
                        x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Experience States', axis=alt.Axis(title="")),
                        color=alt.Color('Agent', scale=color_scale, legend=None), tooltip=['Time', 'Experience States', 'Agent']
                    ).interactive()
                    st.altair_chart(states_chart, use_container_width=True)

            if st.checkbox("Show Raw Data"):
                st.dataframe(combined_data)
        else:
            st.warning("Please select at least one agent from the sidebar to display visualizations.")
    else:
        st.info("⬅️ Run a simulation or upload a pickle file with agent data to begin.")