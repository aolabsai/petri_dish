import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from main_classes import PetriDish


# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="Petri Dish Environment Builder")
st.title("🔬 Petri Dish Environment Builder")
st.write("Interactively design and visualize stimuli distributions for your 2D environment.")

# --- Initialize Session State ---
layer_headers = ["Food Layer", "Chemical-A Layer", "Chemical-B Layer"]
if 'distributions' not in st.session_state:
    st.session_state.distributions = [
        {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0.0, 'max_p': 1.0, 'start_pos': (0.0, 0.0), 'end_pos': (1.0, 1.0)},
        {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0.0, 'max_p': 1.0, 'start_pos': (0.0, 0.0), 'end_pos': (1.0, 1.0)},
        {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0.0, 'max_p': 1.0, 'start_pos': (0.0, 0.0), 'end_pos': (1.0, 1.0)},
    ]
    st.session_state.stimuli_intensity = []
    st.session_state.dish = []

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Global Settings")
    grid_size = st.slider("Grid Size", 50, 500, 200, 10)
    st.session_state.stimuli_intensity = st.slider("Max Stimuli Intensity (n)", 1, 50, 9, 1)
    
    # st.header("Layer Management")
    # if st.button("＋ Add New Layer"):
    #     new_layer = {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0.0, 'max_p': 1.0, 'start_pos': (0.0, 0.0), 'end_pos': (1.0, 1.0)}
    #     st.session_state.distributions.append(new_layer)
        
    # st.header("Export Environment")
    # st.write("Save the current setup list to a pickle file.")
    # if st.session_state.distributions:
    #     export_dish = PetriDish(
    #         size=grid_size, 
    #         distributions=st.session_state.distributions, 
    #         stimuli_intensity=st.session_state.stimuli_intensity
    #     )
        # # Create a deep copy for serialization to avoid including the large 'custom_mask' array
        # export_distributions = []
        # for dist in st.session_state.distributions:
        #     dist_copy = dist.copy()
        #     if 'custom_mask' in dist_copy:
        #         del dist_copy['custom_mask'] # Remove mask before saving
        #     export_distributions.append(dist_copy)

    # if st.session_state.dish:
        # env_bytes = pickle.dumps(export_dish)
        # st.download_button(
        #     label="Download dish_env.pkl",
        #     data=env_bytes,
        #     file_name="dish_env.pkl",
        #     mime="application/octet-stream"
        # )
    if st.session_state.dish:
        if st.button("Save Environment", help="Happy with your petri dish? Click this button to save the env and move on to run your assay.", type="primary", icon="🟢"):
            st.switch_page("pages/2_Step_2_-_Run Assay.py")
    
    # else:
    #     st.info("Add at least one layer to enable export.")


# --- Main Layout ---
if not st.session_state.distributions:
    st.info("⬅️ Add a layer from the sidebar to begin designing your environment.")
else:
    # --- Part 1: Configuration Controls ---
    st.header("Layer Configurations")
    config_cols = st.columns(len(st.session_state.distributions))

    for i, layer_config in enumerate(st.session_state.distributions):
        with config_cols[i]:
            with st.container(border=True):
                st.subheader(layer_headers[i])
                # st.subheader(f"Layer {i}")
                key_prefix = f"layer_{i}"
                layer_config['type'] = st.selectbox("Distribution Type", ['linear', 'radial', 'quadrant', 'custom'], index=['linear', 'radial', 'quadrant', 'custom'].index(layer_config.get('type', 'linear')), key=f"{key_prefix}_type")
                prob_range = st.slider("Probability Range", 0.0, 1.0, (layer_config.get('min_p', 0.0), layer_config.get('max_p', 1.0)), key=f"{key_prefix}_prob")
                layer_config['min_p'], layer_config['max_p'] = prob_range
                st.write("Active Area")
                x_range = st.slider("X-Axis Range", 0.0, 1.0, (layer_config.get('start_pos', (0.0, 0.0))[0], layer_config.get('end_pos', (1.0, 1.0))[0]), key=f"{key_prefix}_x_range")
                y_range = st.slider("Y-Axis Range", 0.0, 1.0, (layer_config.get('start_pos', (0.0, 0.0))[1], layer_config.get('end_pos', (1.0, 1.0))[1]), key=f"{key_prefix}_y_range")
                layer_config['start_pos'], layer_config['end_pos'] = (x_range[0], y_range[0]), (x_range[1], y_range[1])
                if layer_config['type'] == 'linear': layer_config['direction'] = st.selectbox("Direction", ['horizontal-rightleft', 'horizontal-leftright', 'vertical-downup', 'vertical-updown'], key=f"{key_prefix}_linear_dir")
                elif layer_config['type'] == 'radial': layer_config['position'] = st.selectbox("Peak Position", ['center', 'random', 'top-left', 'top-right', 'bottom-left', 'bottom-right'], key=f"{key_prefix}_radial_pos")
                elif layer_config['type'] == 'quadrant':
                    if 'quadrant_setups' not in layer_config: layer_config['quadrant_setups'] = []
                    if st.button("＋ Add Quadrant", key=f"{key_prefix}_add_quad"):
                        layer_config['quadrant_setups'].append({'quadrant': 'top-right', 'peak': 'center', 'diffusion': 'linear'})
                    for j, quad_setup in enumerate(layer_config['quadrant_setups']):
                        quad_key = f"{key_prefix}_quad_{j}"
                        with st.expander(f"Setup {j+1}", expanded=True):
                            quad_setup['quadrant'] = st.selectbox("Quadrant", ['top-left', 'top-right', 'bottom-left', 'bottom-right'], key=f"{quad_key}_quad")
                            quad_setup['peak'] = st.selectbox("Peak", ['center', 'corner'], key=f"{quad_key}_peak")
                            quad_setup['diffusion'] = st.selectbox("Diffusion", ['linear', 'radial'], key=f"{quad_key}_diffusion")
                            if st.button("Delete", key=f"{quad_key}_delete", type="secondary"):
                                layer_config['quadrant_setups'].pop(j)
                                st.rerun()
                elif layer_config['type'] == 'custom':
                    st.write("Draw a mask for the distribution.")
                    stroke_width = st.slider("Brush Size", 1, 50, 20, key=f"{key_prefix}_stroke")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 1)",
                        stroke_width=stroke_width,
                        stroke_color="rgba(0, 0, 0, 1)",
                        background_color="rgba(0, 0, 0, 0)",
                        update_streamlit=True,
                        height=200,
                        width=200,
                        drawing_mode="freedraw",
                        key=f"{key_prefix}_canvas",
                    )
                    if st.button("Apply Drawing", key=f"{key_prefix}_apply_drawing"):
                        if canvas_result.image_data is not None:
                            # Use the alpha channel as the mask for smooth, anti-aliased edges
                            mask_alpha = canvas_result.image_data[:, :, 3].astype(np.float32) / 255.0
                            
                            # Convert to PIL Image to resize it to the main grid size
                            pil_img = Image.fromarray(mask_alpha)
                            resized_pil_img = pil_img.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
                            
                            # Convert back to a numpy array and store in the layer configuration
                            resized_mask = np.array(resized_pil_img)
                            layer_config['custom_mask'] = resized_mask
                            st.success("Drawing has been applied as a mask.")
                            # Rerun to update the visualizations with the new mask
                            st.rerun()

                if st.button("🗑️ Remove Layer", key=f"{key_prefix}_remove", use_container_width=True):
                    st.session_state.distributions.pop(i)
                    st.rerun()

    # --- Generate the dish once after all configs are set ---
    try:
        st.session_state.dish = PetriDish(size=grid_size, distributions=st.session_state.distributions, stimuli_intensity=st.session_state.stimuli_intensity)
        
        # --- Part 2: Individual Layer Visualizations ---
        st.header("Individual Layer Views")
        st.write("Each view below corresponds to the configuration column directly above it.")
        vis_cols = st.columns(len(st.session_state.distributions))
        for i, col in enumerate(vis_cols):
            with col:
                fig_layer = st.session_state.dish.visualize_layer(i)
                if fig_layer:
                    st.pyplot(fig_layer)

        # --- Part 3: Combined View ---
        
        # Create three columns to center the combined view
        st.write("##")
        _left_spacer, center_col, _right_spacer = st.columns([1, 1, 1])
        with center_col:
            st.header("Combined Environment View")
            fig_combined = st.session_state.dish.visualize_combined()
            st.pyplot(fig_combined)

    except ValueError as e:
        st.error(f"Error generating environment: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")