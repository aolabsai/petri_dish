import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AO | Petri Dish Benchmark",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is a petri dish simulation app!"
    # }
)

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload your experiment data",
    type=["pkl"],
    help="Upload a pickle file containing a pandas DataFrame with agent experiment data."
)

st.title("Continuous Learning Benchmark: Petri Dish Simulation")


def create_layer_rgb(stimuli_layer, color_rgb, grid_size):
    """
    Creates an RGB image layer from a 2D binary array of stimuli.

    Args:
        stimuli_layer (np.ndarray): A 2D numpy array where True/1s represent stimuli locations.
        color_rgb (list): The [R, G, B] color for the stimuli.
        grid_size (int): The size of the grid world.

    Returns:
        np.ndarray: A 3D numpy array representing the RGB image layer.
    """
    # Create an empty (black) RGB canvas
    rgb = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    # Efficiently apply the specified color where stimuli are present
    rgb[stimuli_layer.astype(bool)] = color_rgb
    
    return rgb

def plot_grid(layer_rgb, combined_data, agent_colors, selected_agents, grid_size):
    """
    Plots the grid world, agent paths, and their start/end points.
    - Start position is marked with a circle 'o'.
    - End position is marked with an 'X'.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(layer_rgb)

    for agent in selected_agents:
        agent_data = combined_data[combined_data['Agent'] == agent]
        if not agent_data.empty:
            # Plot the agent's full path
            ax.plot(agent_data['pos_x'], agent_data['pos_y'], color=agent_colors[agent], linewidth=2, alpha=0.8)
            
            # Get and plot the start position
            start_pos = agent_data.iloc[0]
            ax.scatter(start_pos['pos_x'], start_pos['pos_y'], 
                       marker='o', s=10, 
                       color=agent_colors[agent], 
                       edgecolor='white', linewidth=0.5, zorder=5)

            # Get and plot the end position
            end_pos = agent_data.iloc[-1]
            ax.scatter(end_pos['pos_x'], end_pos['pos_y'], 
                       marker='X', s=50, 
                       color=agent_colors[agent], 
                       edgecolor='white', linewidth=0.5, zorder=5)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0)
    return fig


# Main application logic starts after a file is successfully uploaded
if uploaded_file is not None:
    try:
        # Load the data from the uploaded pickle file
        all_data = pd.read_pickle(uploaded_file)

        # Unpack the incoming data structure: [[agent_names, agent_dfs], env_data]
        agent_info = all_data[0]
        env_data = all_data[1]

        # Extract agent names and their corresponding dataframes
        agent_names = agent_info[0]
        agent_dfs = agent_info[1]

        # Add the 'Agent' column to each dataframe and prepare for concatenation
        data_to_concat = []
        for i, df in enumerate(agent_dfs):
            # Make a copy to avoid SettingWithCopyWarning
            df_copy = df.copy()
            df_copy['Agent'] = agent_names[i]
            data_to_concat.append(df_copy)
        
        # Combine all agent data into a single dataframe
        full_data = pd.concat(data_to_concat, ignore_index=True)
        
        grid_size = env_data[0].shape[0]

        # --- MODIFICATION: Updated required_cols to check for layer-specific columns ---
        required_cols = {
            'Time', 'Experience States', 'pos_x', 'pos_y', 'Agent',
            'stimuli0', 'stimuli1', 'stimuli2',
            'neuronal0', 'neuronal1', 'neuronal2'
        }
        
        # --- MODIFICATION: Dynamically find learning event columns ---
        learning_event_cols = [col for col in full_data.columns if 'Events' in col]
        if not learning_event_cols:
             # Look for 'Pleasure' and 'Pain' as a fallback
            fallback_cols = [col for col in ['Pleasure', 'Pain'] if col in full_data.columns]
            if fallback_cols:
                learning_event_cols = fallback_cols
            else:
                st.error("Upload Error: No learning event columns (e.g., ending in 'Events' or named 'Pleasure'/'Pain') found in the DataFrame.")
                st.stop()
            
        if not required_cols.issubset(full_data.columns):
            missing_cols = required_cols - set(full_data.columns)
            st.error(f"Upload Error: The DataFrames in the pickle file must contain the following columns: {', '.join(missing_cols)}")
            st.stop()

    except Exception as e:
        st.error(f"An error occurred while reading or processing the pickle file: {e}")
        st.stop()

    # Dynamically get the list of agents from the loaded list of names
    agents = agent_names

    # Agent selection UI in the sidebar
    st.sidebar.header("Agent Selection")
    selected_all_agents = st.sidebar.checkbox("Display all Agents", value=True)

    if selected_all_agents:
        selected_agents = agents
    else:
        selected_agents = st.sidebar.multiselect("Or select Agents to display", agents, default=agents)

    # Filter the data based on the selected agents
    if selected_agents:
        combined_data = full_data[full_data['Agent'].isin(selected_agents)].copy()


        # --- Dynamic and Visualization Section ---

        # Define a consistent color scheme for all possible agents
        color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        agent_colors = {agent: color_list[i % len(color_list)] for i, agent in enumerate(agents)}

        # Import stimuli layers from the loaded env_data
        try:
            stimuli1_layer = env_data[0]
            stimuli2_layer = env_data[1]
            stimuli3_layer = env_data[2]
        except (IndexError, TypeError) as e:
            st.error(f"Could not read stimuli layers from the uploaded file. Ensure 'env_data' has the correct format. Error: {e}")
            st.stop()

        # Create the RGB layers for visualization
        layer1_rgb = create_layer_rgb(stimuli1_layer, [255, 0, 0], grid_size)  # Red
        layer2_rgb = create_layer_rgb(stimuli2_layer, [0, 255, 0], grid_size)  # Green
        layer3_rgb = create_layer_rgb(stimuli3_layer, [0, 0, 255], grid_size)  # Blue
        
        # Combine layers for the composite view
        layer4_rgb = np.clip(layer1_rgb.astype(np.uint16) + layer2_rgb.astype(np.uint16) + layer3_rgb.astype(np.uint16), 0, 255).astype(np.uint8)

        # Display the agent legend
        legend_text = " **Agent Legend:** " + " | ".join([
            f"<span style='color:{agent_colors[agent]}; font-weight:bold;'>{agent}</span>"
            for agent in selected_agents
        ])
        st.markdown(legend_text, unsafe_allow_html=True)

        # Color scale for Altair charts
        color_scale = alt.Scale(domain=list(agent_colors.keys()), range=list(agent_colors.values()))

        st.header("Agent Movement Over Time")

        with st.expander("Environment View (Combined)", expanded=True):
            st.pyplot(plot_grid(layer4_rgb, combined_data, agent_colors, selected_agents, grid_size))

        with st.expander("Env Views by Stimuli (Filtered)", expanded=False):
            grid_col1, grid_col2, grid_col3 = st.columns(3)
            with grid_col1:
                st.subheader("Layer 1: Red (Food)")
                st.pyplot(plot_grid(layer1_rgb, combined_data, agent_colors, selected_agents, grid_size))
            with grid_col2:
                st.subheader("Layer 2: Green")
                st.pyplot(plot_grid(layer2_rgb, combined_data, agent_colors, selected_agents, grid_size))
            with grid_col3:
                st.subheader("Layer 3: Blue")
                st.pyplot(plot_grid(layer3_rgb, combined_data, agent_colors, selected_agents, grid_size))

        st.header("Agent Activity Over Time")
        
        # --- MODIFICATION: Create 3 graphs for Stimuli and Neuronal responses ---
        with st.expander("I (input) Neuron Activations by Stimuli Layer", expanded=True):
            s_col1, s_col2, s_col3 = st.columns(3)
            layer_titles = ["Layer 1: Red (Food)", "Layer 2: Green", "Layer 3: Blue"]
            stimuli_cols = ['stimuli0', 'stimuli1', 'stimuli2']
            
            for i, col in enumerate([s_col1, s_col2, s_col3]):
                with col:
                    st.subheader(layer_titles[i])
                    response_chart = alt.Chart(combined_data).mark_line().encode(
                        x=alt.X('Time', axis=alt.Axis(title=None)), 
                        y=alt.Y(stimuli_cols[i], axis=alt.Axis(title=None)),
                        color=alt.Color('Agent', scale=color_scale, legend=None),
                        tooltip=['Time', stimuli_cols[i], 'Agent']
                    ).interactive()
                    st.altair_chart(response_chart, use_container_width=True)

        with st.expander("Q (inner) Neuron Activations by Stimuli Layer", expanded=True):
            n_col1, n_col2, n_col3 = st.columns(3)
            neuronal_cols = ['neuronal0', 'neuronal1', 'neuronal2']

            for i, col in enumerate([n_col1, n_col2, n_col3]):
                with col:
                    st.subheader(layer_titles[i])
                    neuronal_chart = alt.Chart(combined_data).mark_line().encode(
                        x=alt.X('Time', axis=alt.Axis(title=None)), 
                        y=alt.Y(neuronal_cols[i], axis=alt.Axis(title=None)),
                        color=alt.Color('Agent', scale=color_scale, legend=None),
                        tooltip=['Time', neuronal_cols[i], 'Agent']
                    ).interactive()
                    st.altair_chart(neuronal_chart, use_container_width=True)
        
        st.markdown("---") # Visual separator

        with st.expander("Learning Events and Memory", expanded=True):
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Control Events")
                st.text("Total number of learning events over time.")
                
                # Check if the specific "Pleasure" and "Pain" case exists
                is_pleasure_pain_case = (len(learning_event_cols) == 2 and set(learning_event_cols) == {'Pleasure', 'Pain'})
                
                combine_pleasure_pain = False
                if is_pleasure_pain_case:
                    combine_pleasure_pain = st.checkbox("View combined plot (Pleasure - Pain)")
                
                # If the checkbox is ticked, show the combined plot
                if combine_pleasure_pain:
                    combined_data['Combined_Events'] = combined_data['Pleasure'] - combined_data['Pain']
                    
                    st.text("Displaying the net of Pleasure minus Pain.")
                    
                    combined_chart = alt.Chart(combined_data).mark_line().encode(
                        x=alt.X('Time', axis=alt.Axis(title="")),
                        y=alt.Y('Combined_Events', axis=alt.Axis(title="Net Events")),
                        color=alt.Color('Agent', scale=color_scale, legend=None),
                        tooltip=['Time', 'Agent', 'Combined_Events']
                    ).interactive()
                    
                    st.altair_chart(combined_chart, use_container_width=True)
                
                # Otherwise, show the original multiselect dropdown for individual event types
                else:
                    # Dropdown for selecting event types
                    selected_event_types = st.multiselect(
                        "Select control event types to display:",
                        options=learning_event_cols,
                        default=learning_event_cols
                    )

                    # Define line styles for different event types
                    line_styles = [[1,0], [8, 4], [3, 3, 2, 2], [8, 4, 2, 4], [2,2]] # Solid, Dashed, Dotted, etc.
                    style_map = {event: line_styles[i % len(line_styles)] for i, event in enumerate(learning_event_cols)}
                    
                    # Display the legend for line styles
                    legend_items = []
                    style_repr = {
                        str([1,0]): "Solid", str([8,4]): "Dashed", str([3,3,2,2]): "Dot-Dash",
                        str([8,4,2,4]): "Long-Dash", str([2,2]): "Dotted"
                    }
                    for event in selected_event_types:
                        style_key = str(style_map.get(event, [1,0]))
                        style_name = style_repr.get(style_key, style_key)
                        legend_items.append(f"<b>{style_name}</b>: {event}")
                    
                    if legend_items:
                        st.markdown(" | ".join(legend_items), unsafe_allow_html=True)

                    if selected_event_types:
                        # Reshape data from wide to long format for Altair
                        learning_data_long = combined_data.melt(
                            id_vars=['Time', 'Agent'],
                            value_vars=selected_event_types,
                            var_name='Event Type',
                            value_name='Count'
                        )

                        # Create the chart
                        learning_chart = alt.Chart(learning_data_long).mark_line(interpolate='step-after').encode(
                            x=alt.X('Time', axis=alt.Axis(title="")),
                            y=alt.Y('Count', axis=alt.Axis(title="Total Events")),
                            color=alt.Color('Agent', scale=color_scale, legend=None),
                            strokeDash=alt.StrokeDash(
                                'Event Type',
                                scale=alt.Scale(domain=list(style_map.keys()), range=list(style_map.values())),
                                legend=None # Hide Altair's default legend
                            ),
                            tooltip=['Time', 'Agent', 'Event Type', 'Count']
                        ).interactive()
                        
                        st.altair_chart(learning_chart, use_container_width=True)
                    else:
                        st.warning("Please select at least one event type to display the chart.")

            with col4:
                st.write("##")
                st.write("#")
                st.subheader("Experience States")
                st.text("Total number of unique memories in the output neuron (unique learning events).")
                states_chart = alt.Chart(combined_data).mark_line().encode(
                    x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Experience States', axis=alt.Axis(title="")),
                    color=alt.Color('Agent', scale=color_scale, legend=None),
                    tooltip=['Time', 'Experience States', 'Agent']
                ).interactive()
                st.altair_chart(states_chart, use_container_width=True)
        # --- MODIFICATION END ---

        if st.checkbox("Show Raw Data"):
            st.dataframe(combined_data)

    else:
        st.warning("Please select at least one agent from the sidebar to display the visualizations.")

# Message to show when no file has been uploaded yet
else:
    st.info("⬅️ Please upload a pickle file with agent data using the sidebar to begin.")