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

st.title("Continuous Learning Benchmark: Petri Dish Simulation")

# --- Initialize Session State ---
if 'saved_dish' not in st.session_state:
    st.session_state.saved_dish = []

if not st.session_state.saved_dish:
    st.info("⬅️ Go to the env page to set an env first.")
    st.page_link("pages/1__Step_1_-_Create_Environment.py", label=" Set your environment.")
else:

    with st.expander("Run a New Simulation", expanded=True):
        st.markdown("""
        Configure and run a new simulation. The results will be displayed in the views below.
        """)
        debug_mode_checkbox = st.checkbox("Enable debug mode", value=False, help='If checked, agents will move randomly (sets agent_archs="random").')
        reuse_agents_from_assay = st.checkbox("Reuse agents from previous assay", value=False, help="If checked, assay will be run with agents from previous trial. WNN agent are natively stateful with memories are persistent across assays.", disabled="assay" not in st.session_state)
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            start_logic_input = st.selectbox("Agent Starting Positions", options=['random', 'center', 'cardinal', 'quadrants', 'corners'], index=0, help="Choose the initial placement pattern for the agents.")
        with sim_col2:
            steps_input = st.slider("Simulation Steps", min_value=1, max_value=1000, value=10, help="Set the total number of time steps for the simulation.")

    # --- New Section: Configure Agent Sensory Range ---
    with st.expander("Configure Agent Sensory Range", expanded=True):
        
        num_agents_input = st.slider("Number of Agents", min_value=1, max_value=100, value=10, help="Select the number of agents to include in the simulation.", disabled=reuse_agents_from_assay)
        
        st.markdown("""
        ---
        Define how agents perceive their immediate environment using different sensory field shapes and extents.
        """)
        
        sensory_shape_input = st.selectbox(
            "Sensory Shape",
            options=['square', 'circle', 'diamond', 'triangle'],
            index=0,
            help="Choose the geometric shape of the agent's sensory field."
        )
        
        sensory_field_col1, sensory_field_col2 = st.columns(2) 

        
        # Determine if horizontal radius slider should be disabled
        disable_horizontal_radius = (sensory_shape_input in ['square', 'circle'])


        with sensory_field_col1:

            sensory_col1, sensory_col2 = st.columns(2)
    
            with sensory_col1:
                sensory_radius_vertical = st.slider(
                    "Vertical Sensory Radius (Rv)",
                    min_value=1,
                    max_value=5,
                    value=1,
                    help="Set the vertical radius of the sensory field (in grid points). For square/circle, this defines the overall radius."
                )
            
            with sensory_col2:
                sensory_radius_horizontal = st.slider(
                    "Horizontal Sensory Radius (Rh)",
                    min_value=1,
                    max_value=5,
                    value=sensory_radius_vertical, # Default to vertical radius if disabled
                    disabled=disable_horizontal_radius,
                    help="Set the horizontal radius of the sensory field (in grid points). Only applicable for 'diamond' and 'triangle' shapes."
                )
            
            # For square/circle, ensure horizontal radius matches vertical for consistency in underlying logic
            if disable_horizontal_radius:
                sensory_radius_horizontal = sensory_radius_vertical

            # Calculate and display Encephalization Quotient
            if sensory_radius_vertical > 0:
                encephalization_quotient = sensory_radius_horizontal / sensory_radius_vertical
                st.metric(label="Encephalization Quotient (Rh / Rv)", value=f"{encephalization_quotient:.2f}",
                        help="Ratio of horizontal sensory radius to vertical sensory radius. Indicates the relative 'width' of perception compared to 'height'.")
            else:
                st.metric(label="Encephalization Quotient (Rh / Rv)", value="N/A",
                        help="Ratio of horizontal sensory radius to vertical sensory radius. Indicates the relative 'width' of perception compared to 'height'.")


        with sensory_field_col2:

            st.subheader("Visualizing Sensory Input (for 1 stimuli layer)")
            st.markdown("This visualization shows what an agent's sensory input would look like for a single layer based on your current settings, with the agent at the center.")

            # Create a blank grid for visualization
            max_radius = max(sensory_radius_vertical, sensory_radius_horizontal)
            grid_size_viz = (max_radius * 2) + 1
            sensory_grid = np.zeros((grid_size_viz, grid_size_viz))
            center_x, center_y = max_radius, max_radius # Agent at the center of the visualization grid
            
            st.session_state.sensory_field_layer_size = 0
            # Populate sensory grid based on shape and radii
            for i in range(grid_size_viz):
                for j in range(grid_size_viz):
                    is_in_sensory_field = False
                    # Relative coordinates from the agent's center
                    dy_rel = i - center_x # vertical offset
                    dx_rel = j - center_y # horizontal offset

                    if sensory_shape_input == 'square':
                        if abs(dy_rel) <= sensory_radius_vertical and abs(dx_rel) <= sensory_radius_vertical:
                            is_in_sensory_field = True
                    elif sensory_shape_input == 'circle':
                        if (dy_rel**2 + dx_rel**2) <= sensory_radius_vertical**2:
                            is_in_sensory_field = True
                    elif sensory_shape_input == 'diamond':
                        # L1 norm based on vertical and horizontal radii
                        # abs(dy_rel)/Rv + abs(dx_rel)/Rh <= 1.0
                        if (sensory_radius_vertical > 0 and sensory_radius_horizontal > 0) and \
                        (abs(dy_rel) / sensory_radius_vertical + abs(dx_rel) / sensory_radius_horizontal <= 1.0):
                            is_in_sensory_field = True
                    elif sensory_shape_input == 'triangle':
                        # Triangle pointing "up" (towards smaller i values on the grid, if agent faces that way)
                        # Apex: (center_x - Rv, center_y)
                        # Base: at i = center_x + Rv, from center_y - Rh to center_y + Rh
                        
                        # Check vertical bounds
                        if dy_rel >= -sensory_radius_vertical and dy_rel <= sensory_radius_vertical:
                            # Calculate half-width at current vertical offset
                            # y_rel from apex is dy_rel + Rv (ranges from 0 to 2*Rv)
                            y_from_apex_proportional = (dy_rel + sensory_radius_vertical) / (2 * sensory_radius_vertical + 1e-9) # Add small epsilon to avoid div by zero
                            
                            current_half_width = sensory_radius_horizontal * y_from_apex_proportional
                            
                            if abs(dx_rel) <= current_half_width:
                                is_in_sensory_field = True
                    
                    if is_in_sensory_field:
                        sensory_grid[i, j] = 0.5 # Indicate sensory field
                        st.session_state.sensory_field_layer_size += 1
            
            # Mark the agent's center
            sensory_grid[center_x, center_y] = 1.0 # Brighter to indicate agent's position

            # Plotting the sensory grid
            fig_sensory, ax_sensory = plt.subplots(figsize=(3, 3))
            ax_sensory.imshow(sensory_grid, cmap='viridis', origin='upper', vmin=0, vmax=1)
            ax_sensory.set_xticks(np.arange(-.5, grid_size_viz, 1), minor=True)
            ax_sensory.set_yticks(np.arange(-.5, grid_size_viz, 1), minor=True)
            ax_sensory.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
            ax_sensory.set_xticks([])
            ax_sensory.set_yticks([])
            ax_sensory.set_title(f"Agent Sensory Field ({sensory_shape_input.capitalize()}, Rv:{sensory_radius_vertical}, Rh:{sensory_radius_horizontal})")
            st.pyplot(fig_sensory)
            plt.close(fig_sensory)

    with st.expander("Set Agent-level hyperparameters:", expanded=False):
        st.markdown("""
        Neurons' lookup tables maintain a record of how often each row is used for inference during CGA (in ao.agent.neuron.c_info). This information is used to prune lookup tables so that stale, unused rows are deleted, the desired emergent (or agent-level affect) being the "forgetting" component of classical conditioning. Set those parameters below:
        """)
        

        sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
        with sim_col1:
            st.session_state.instincts_checkbox = st.checkbox("Enable instincts", value=True, help='Disabling this will disable new learning')
            st.write("The agent sums the inputs (per layer) of the immediate points around it (9 points total, including its own position); food-related pain/pleasure instincts can be set up to activate based on a threshold percentage of the input.")
            if st.session_state.instincts_checkbox:
                in_col1, in_col2 = st.columns(2)
                with in_col1:
                    st.session_state.pain_threshold = st.number_input("Threshold for pain:", min_value=0.0, max_value=1.0, value=1/3)
                with in_col2:
                    st.session_state.pleasure_threshold = st.number_input("Threshold for pleasure:", min_value=0.0, max_value=1.0, value=2/3)

        with sim_col2:
            save_C_info = st.checkbox("Enable neuron-level pruning for agent-level forgetting", value=False, help="")
            if save_C_info: st.warning('This is an experimental feature still in development; to use it be sure to run on [this branch](https://github.com/aolabsai/ao_core/tree/research_expansion/neuron_pruning) of ao_core', icon="⚠️")
            C_impression_initial = st.number_input("Initial learning experience impression strength", min_value=1, max_value=100, value=5, help="The number added to the memory'", disabled=not save_C_info) # strength of impression when first added to neuron from C learning event
            C_impression_match = st.number_input("Impression increment if lookup match", min_value=1, max_value=C_impression_initial, value=2, help="The number incremented when there is a lookup match", disabled=not save_C_info)  # increment of impression if accessed by neuron during inference
            C_pruning = st.number_input("Impression decrement for all other rows that did not match", min_value=1, max_value=C_impression_match, value=1, help="The number decremented when no lookup match", disabled=not save_C_info) # decrement of impression in C_info if not accessed by neuron during inference
            C_pruning_cutoff = st.number_input("Number below which memories are deleted from neurons' lookup tables", min_value=1, max_value=100, value=1, help="Cutoff value below which memories are deleted", disabled=not save_C_info) # value below which impressions are pruned from neuron)

        with sim_col3:
            saturation_refractory = st.checkbox("Enable saturation and refractory limits to neural responses", value=False, help="")
            if saturation_refractory: st.warning('This is an experimental feature still in development; to use it be sure to run on [this branch](https://github.com/aolabsai/ao_core/tree/research_expansion/neuron_pruning) of ao_core', icon="⚠️")
            saturation_threshold = st.number_input("Number of consecutively repeated neuron outputs before the opposite output is forced (eg. set this value to 5; if a neuron responds `1` 5 times, then force its response to be `0` for the 6th time)", min_value=0, max_value=20, value=10, help="", disabled=not saturation_refractory) # strength of impression when first added to neuron from C learning event
            refractory_period = st.number_input("Number of states before a response-saturated neuron once again responds regularly", min_value=0, max_value=20, value=5, help="", disabled=not saturation_refractory)  # increment of impression if accessed by neuron during inference

        with sim_col4:
            pretrain_agents = st.checkbox("Pre-train agents on random data", help="Helpful to increase agent's response variance by pre-populating agent neurons' lookup tables, giving a wider range before lessons from learning kick in and override.")
            if pretrain_agents: pretrain_agent_steps = st.number_input("Add random states to agents", min_value=0, max_value=100, value=10)


    if st.button("▶️ Run Simulation"):
        with st.spinner(f"Running simulation for {steps_input} steps..."):
            petri_dish = st.session_state.saved_dish
            agent_archs_param = "random" if debug_mode_checkbox else updateArch(st.session_state.sensory_field_layer_size, st.session_state.stimuli_intensity, st.session_state.pain_threshold, st.session_state.pleasure_threshold) # the arch that is imported from the "archs" folder
            
            if reuse_agents_from_assay and "assay" in st.session_state:
                assay_loadagents = st.session_state.assay
            else:
                assay_loadagents = ""
            # Pass sensory parameters to Assay
            assay = Assay(
                petri_dish=petri_dish,
                num_agents=num_agents_input,
                start_logic=start_logic_input,
                agent_archs=agent_archs_param,
                steps=steps_input,
                assay_loadagents=assay_loadagents,
                sensory_shape=sensory_shape_input,
                sensory_radius_vertical=sensory_radius_vertical,
                sensory_radius_horizontal=sensory_radius_horizontal
            )
            
            if save_C_info: assay.set_agent_hyperparameters(
                C_impression_initial, # strength of impression when first added to neuron from C learning event
                C_impression_match, # increment of impression if accessed by neuron during inference
                C_pruning, # decrement of impression in C_info if not accessed by neuron during inference
                C_pruning_cutoff, # value below which impressions are pruned from neuron)
                save_C_info, # whether or not to disable this feature
                saturation_threshold,
                refractory_period,
                saturation_refractory
                )

            if pretrain_agents: assay.pretrain_random(pretrain_agent_steps)

            assay.INSTINCTS = st.session_state.instincts_checkbox # to activate training, let's gooooo
            assay.run_step(steps=steps_input)
            st.session_state.assay = assay
            st.session_state.assay_saved = assay.export_data()
            st.success("Simulation complete! Results are displayed below.")
            st.rerun()


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

    if 'assay_saved' in st.session_state:
        try:
            agent_data, env_data = st.session_state.assay.agent_data, st.session_state.assay.env_data
            agent_names, agent_dfs = agent_data[0], agent_data[1]
            
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
        else:
            st.warning("Please select at least one agent from the sidebar to display visualizations.")

        with st.sidebar:
            st.markdown("---")
            if "assay" in st.session_state:
                st.write("Finished? Export experiment data as a pickle file.")
                file_name = st.text_input("Experiment file name:", value="exp01")                
                import dill as pickle
                assay_bytes = pickle.dumps(st.session_state.assay)
                st.download_button(
                    label="Save experiment to .pkl file.",
                    data=assay_bytes,
                    file_name= file_name+".pkl",
                    mime="application/octet-stream"
                )

        num_layers = len(env_data)
        # Generate RGB data for each layer and the combined view
        max_intensity = float(max(layer.max() for layer in env_data if layer.size > 0) or 1)
        layer1_rgb = visualize_layer_data(env_data[0], 0, num_layers, max_intensity)
        layer2_rgb = visualize_layer_data(env_data[1], 1, num_layers, max_intensity)
        layer3_rgb = visualize_layer_data(env_data[2], 2, num_layers, max_intensity)
        layer4_rgb = visualize_combined_data(env_data, max_intensity)

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
                st.subheader("Instinct Activations (control events)")
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
                st.subheader("Lookup-table Rows")
                st.text("Total unique memories in the output neuron.")
                states_chart = alt.Chart(combined_data).mark_line().encode(
                    x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Experience States', axis=alt.Axis(title="")),
                    color=alt.Color('Agent', scale=color_scale, legend=None), tooltip=['Time', 'Experience States', 'Agent']
                ).interactive()
                st.altair_chart(states_chart, use_container_width=True)

        if st.checkbox("Show Raw Data"):
            st.dataframe(combined_data)
    else:
        st.info("⬅️ Run a simulation or upload a pickle file with agent data to begin.")