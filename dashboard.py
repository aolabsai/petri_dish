import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Petri Dish Experiment Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is a petri dish simulation app!"
    }
)

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload your experiment data",
    type=["pkl"],
    help="Upload a pickle file containing a pandas DataFrame with agent experiment data."
)

st.title("Agent Performance Dashboard in 2D Grid World")


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

        full_data = all_data[0]
        env_data = all_data[1]
        grid_size = env_data[0].shape[0]

        # Validate that the necessary columns exist
        required_cols = {'Time', 'Response to Stimuli', 'Neuronal Response', 'Learning Events', 'Experience States', 'pos_x', 'pos_y', 'Agent'}
        if not required_cols.issubset(full_data.columns):
            st.error(f"Upload Error: The DataFrame in the pickle file must contain the following columns: {', '.join(required_cols)}")
            st.stop()

    except Exception as e:
        st.error(f"An error occurred while reading the pickle file: {e}")
        st.stop()

    # Dynamically get the list of agents from the dataframe
    agents = sorted(full_data['Agent'].unique())

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

        with st.expander("Grid World Views", expanded=True):
            grid_col1, grid_col2, grid_col3, grid_col4 = st.columns(4)
            with grid_col1:
                st.subheader("Layer 1: Red Stimuli")
                st.pyplot(plot_grid(layer1_rgb, combined_data, agent_colors, selected_agents, grid_size))
            with grid_col2:
                st.subheader("Layer 2: Green Stimuli")
                st.pyplot(plot_grid(layer2_rgb, combined_data, agent_colors, selected_agents, grid_size))
            with grid_col3:
                st.subheader("Layer 3: Blue Stimuli")
                st.pyplot(plot_grid(layer3_rgb, combined_data, agent_colors, selected_agents, grid_size))
            with grid_col4:
                st.subheader("Combined Layers")
                st.pyplot(plot_grid(layer4_rgb, combined_data, agent_colors, selected_agents, grid_size))

        with st.expander("Time Series Charts", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            # (Chart rendering code remains the same as your original)
            with col1:
                st.subheader("Response to Stimuli")
                response_chart = alt.Chart(combined_data).mark_line().encode(
                    x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Response to Stimuli', axis=alt.Axis(title="")),
                    color=alt.Color('Agent', scale=color_scale, legend=None),
                    tooltip=['Time', 'Response to Stimuli', 'Agent']
                ).interactive()
                st.altair_chart(response_chart, use_container_width=True)
            with col2:
                st.subheader("Neuronal Response")
                neuronal_chart = alt.Chart(combined_data).mark_line().encode(
                    x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Neuronal Response', axis=alt.Axis(title="")),
                    color=alt.Color('Agent', scale=color_scale, legend=None),
                    tooltip=['Time', 'Neuronal Response', 'Agent']
                ).interactive()
                st.altair_chart(neuronal_chart, use_container_width=True)
            with col3:
                st.subheader("Control Events")
                learning_chart = alt.Chart(combined_data).mark_line(interpolate='step-after').encode(
                    x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Learning Events', axis=alt.Axis(title="")),
                    color=alt.Color('Agent', scale=color_scale, legend=None),
                    tooltip=['Time', 'Learning Events', 'Agent']
                ).interactive()
                st.altair_chart(learning_chart, use_container_width=True)
            with col4:
                st.subheader("Experience States")
                states_chart = alt.Chart(combined_data).mark_line().encode(
                    x=alt.X('Time', axis=alt.Axis(title="")), y=alt.Y('Experience States', axis=alt.Axis(title="")),
                    color=alt.Color('Agent', scale=color_scale, legend=None),
                    tooltip=['Time', 'Experience States', 'Agent']
                ).interactive()
                st.altair_chart(states_chart, use_container_width=True)

        if st.checkbox("Show Raw Data"):
            st.dataframe(combined_data)

    else:
        st.warning("Please select at least one agent from the sidebar to display the visualizations.")

# Message to show when no file has been uploaded yet
else:
    st.info("⬅️ Please upload a pickle file with agent data using the sidebar to begin.")