import streamlit as st
import pandas as pd
import numpy as np
import altair as alt  # For interactive charts
import matplotlib.pyplot as plt  # For grid visualizations

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Title of the dashboard
st.title("Agent Performance Dashboard in 2D Grid World")

# Sample list of agents
agents = ['Agent1', 'Agent2', 'Agent3']


# Multi-select for agents to display (default to all)
selected_all_agents = st.checkbox("Display all Agents")

if selected_all_agents:
    selected_agents = agents
else:
    selected_agents = st.multiselect("Or select Agents to display", agents, default=agents)

# Define a consistent color scheme for agents
color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
agent_colors = {agent: color_list[i % len(color_list)] for i, agent in enumerate(agents)}

# Grid size for the 2D world
grid_size = 50

# Generate dummy stimuli positions independently for each layer (allowing overlaps)
num_stimuli_per_layer = 3
stimuli1 = np.random.choice(np.arange(grid_size ** 2), num_stimuli_per_layer, replace=False)
stimuli2 = np.random.choice(np.arange(grid_size ** 2), num_stimuli_per_layer, replace=False)
stimuli3 = np.random.choice(np.arange(grid_size ** 2), num_stimuli_per_layer, replace=False)

# Function to create RGB layer
def create_layer_rgb(stimuli_list, color_rgb, grid_size):
    rgb = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    for s in stimuli_list:
        x = s % grid_size
        y = s // grid_size
        rgb[y, x] = color_rgb
    return rgb

# Create layers
layer1_rgb = create_layer_rgb(stimuli1, [255, 0, 0], grid_size)  # Red
layer2_rgb = create_layer_rgb(stimuli2, [0, 255, 0], grid_size)  # Green
layer3_rgb = create_layer_rgb(stimuli3, [0, 0, 255], grid_size)  # Blue
layer4_rgb = np.clip(layer1_rgb.astype(np.uint16) + layer2_rgb.astype(np.uint16) + layer3_rgb.astype(np.uint16), 0, 255).astype(np.uint8)  # Combined with clipping to avoid overflow

# Function to plot grid with agent paths
def plot_grid(layer_rgb, combined_data, agent_colors, selected_agents, grid_size):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(layer_rgb)
    for agent in selected_agents:
        agent_data = combined_data[combined_data['Agent'] == agent]
        ax.plot(agent_data['pos_x'], agent_data['pos_y'], color=agent_colors[agent], linewidth=2, alpha=0.8)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color='white', linewidth=0)
    ax.tick_params(axis='both', which='both', length=0)
    return fig

# Function to generate dummy data for demonstration
def generate_dummy_data(time_steps=100, grid_size=4):
    time = np.arange(time_steps)
    # Response to stimuli: random walk-like data
    response = np.random.randn(time_steps).cumsum()
    # Neuronal response: simulated neuron activations
    neuronal = np.sin(time / 10) + np.random.normal(0, 0.5, time_steps)
    # Learning events: sparse events (e.g., 1 when learning occurs)
    learning = np.random.choice([0, 1], size=time_steps, p=[0.9, 0.1])
    
    # Simulate 2D random walk for positions
    pos_x = np.zeros(time_steps, dtype=int)
    pos_y = np.zeros(time_steps, dtype=int)
    current_x = current_y = grid_size // 2
    for t in range(time_steps):
        pos_x[t] = current_x
        pos_y[t] = current_y
        dx, dy = np.random.choice([-1, 0, 1], 2)
        current_x = np.clip(current_x + dx, 0, grid_size - 1)
        current_y = np.clip(current_y + dy, 0, grid_size - 1)
    
    # Experience states: flattened position
    states = pos_y * grid_size + pos_x
    
    return pd.DataFrame({
        'Time': time,
        'Response to Stimuli': response,
        'Neuronal Response': neuronal,
        'Learning Events': learning,
        'Experience States': states,
        'pos_x': pos_x,
        'pos_y': pos_y
    })

if selected_agents:
    # Generate combined data for selected agents
    combined_data = pd.DataFrame()
    for agent in selected_agents:
        data = generate_dummy_data(grid_size=grid_size)
        data['Agent'] = agent
        combined_data = pd.concat([combined_data, data])

    # Display the legend on its own row
    legend_text = " **Agent Legend:** " + " | ".join([
        f"<span style='color:{agent_colors[agent]}; font-weight:bold;'>{agent}</span>"
        for agent in selected_agents
    ])
    st.markdown(legend_text, unsafe_allow_html=True)

    # Color scale for charts
    color_scale = alt.Scale(domain=list(agent_colors.keys()), range=list(agent_colors.values()))

    # Grid world views row (collapsible)
    with st.expander("Grid World Views", expanded=True):
        grid_col1, grid_col2, grid_col3, grid_col4 = st.columns(4)

        with grid_col1:
            st.subheader("Layer 1: Red Stimuli")
            fig1 = plot_grid(layer1_rgb, combined_data, agent_colors, selected_agents, grid_size)
            st.pyplot(fig1)

        with grid_col2:
            st.subheader("Layer 2: Green Stimuli")
            fig2 = plot_grid(layer2_rgb, combined_data, agent_colors, selected_agents, grid_size)
            st.pyplot(fig2)

        with grid_col3:
            st.subheader("Layer 3: Blue Stimuli")
            fig3 = plot_grid(layer3_rgb, combined_data, agent_colors, selected_agents, grid_size)
            st.pyplot(fig3)

        with grid_col4:
            st.subheader("Combined Layers")
            fig4 = plot_grid(layer4_rgb, combined_data, agent_colors, selected_agents, grid_size)
            st.pyplot(fig4)

    # Time series charts row (collapsible)
    with st.expander("Time Series Charts", expanded=True):

        # Create columns for horizontal layout
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Response to Stimuli")
            st.write("Cumulative response to stimuli.")
            response_chart = alt.Chart(combined_data).mark_line().encode(
                x='Time',
                y='Response to Stimuli',
                color=alt.Color('Agent', scale=color_scale, legend=None),
                tooltip=['Time', 'Response to Stimuli', 'Agent']
            ).interactive()
            st.altair_chart(response_chart, use_container_width=True)

        with col2:
            st.subheader("Neuronal Response")
            st.write("Simulated neuronal activations.")
            neuronal_chart = alt.Chart(combined_data).mark_line().encode(
                x='Time',
                y='Neuronal Response',
                color=alt.Color('Agent', scale=color_scale, legend=None),
                tooltip=['Time', 'Neuronal Response', 'Agent']
            ).interactive()
            st.altair_chart(neuronal_chart, use_container_width=True)

        with col3:
            st.subheader("Learning Events")
            st.write("Learning events (1 = event).")
            learning_chart = alt.Chart(combined_data).mark_line(interpolate='step-after').encode(
                x='Time',
                y='Learning Events',
                color=alt.Color('Agent', scale=color_scale, legend=None),
                tooltip=['Time', 'Learning Events', 'Agent']
            ).interactive()
            st.altair_chart(learning_chart, use_container_width=True)

        with col4:
            st.subheader("Experience States")
            st.write("States experienced over time.")
            states_chart = alt.Chart(combined_data).mark_line().encode(
                x='Time',
                y='Experience States',
                color=alt.Color('Agent', scale=color_scale, legend=None),
                tooltip=['Time', 'Experience States', 'Agent']
            ).interactive()
            st.altair_chart(states_chart, use_container_width=True)


    # Optional: Display raw data table for all selected agents
    if st.checkbox("Show Raw Data"):
        st.dataframe(combined_data)
else:
    st.info("Please select at least one agent to display the views.")