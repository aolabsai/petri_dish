import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.colors import ListedColormap
import io
import pickle
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- PetriDish Class Definition ---
# The class is updated with a corrected visualize_combined method.

class PetriDish:
    def __init__(self, size=100, num_layers=3, distributions=None, stimuli_intensity=9):
        """
        Initialize the PetriDish simulation.

        Parameters:
        - size: int, the size of the square grid (size x size).
        - num_layers: int, the number of layers (each representing a different stimulus).
        - distributions: list of dicts, parameters for the distribution of each layer.
          Each dict can have:
            - 'type': 'linear', 'radial', 'quadrant', or 'custom'.
            - 'start_pos': tuple (x, y), optional. The top-left corner of the active distribution area (range [0,1]). Defaults to (0,0).
            - 'end_pos': tuple (x, y), optional. The bottom-right corner of the active distribution area (range [0,1]). Defaults to (1,1).
            - For 'linear':
              - 'direction': str, 'horizontal-rightleft', 'horizontal-leftright', 'vertical-downup', 'vertical-updown'.
              - 'min_p', 'max_p': float, probability range for the gradient.
            - For 'radial':
              - 'position': str or tuple, determines the location of the highest concentration within the active area.
              - 'min_p', 'max_p': float, probability range for the gradient.
            - For 'quadrant':
              - Can be a single setup or a list in 'quadrant_setups'.
              - 'quadrant': str, one of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
              - 'peak': str, 'center' or 'corner' of the quadrant.
              - 'diffusion': str, 'linear' or 'radial'.
              - 'min_p', 'max_p': float, probability range for the gradient.
            - For 'custom':
              - 'custom_mask': A numpy array representing the user's drawing, resized to the grid size.
        - stimuli_intensity: int (default 9), the maximum intensity of a stimulus at a point (range from 0 to n).
        """
        if distributions is None:
            distributions = [
                {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0.0, 'max_p': 0.5}
            ] * num_layers

        self.size = size
        self.distributions = distributions
        self.num_layers = len(self.distributions)
        self.stimuli_intensity = stimuli_intensity
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        self.X, self.Y = np.meshgrid(x, y)
        self.layers = []
        self.generate_layers() # Generate layers upon initialization

    def generate_layers(self):
        """Generates the stimuli layers based on the distributions."""
        self.layers = []
        for d in self.distributions:
            p_grid = self._get_p_grid(d)
            grid = np.random.binomial(self.stimuli_intensity, p_grid).astype(int)
            self.layers.append(grid)

    def _get_position(self, dist):
        if 'position' not in dist:
            return dist.get('center', (0.5, 0.5))
        
        pos = dist['position']
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            cx, cy = map(float, pos)
        elif isinstance(pos, str):
            pos_lower = pos.lower()
            if pos_lower == 'center': cx, cy = 0.5, 0.5
            elif pos_lower == 'random': cx, cy = np.random.uniform(0, 1), np.random.uniform(0, 1)
            elif pos_lower == 'top-left': cx, cy = 0.0, 0.0
            elif pos_lower == 'top-right': cx, cy = 1.0, 0.0
            elif pos_lower == 'bottom-left': cx, cy = 0.0, 1.0
            elif pos_lower == 'bottom-right': cx, cy = 1.0, 1.0
            else: raise ValueError(f"Unknown position string: {pos}")
        else: raise ValueError("position must be tuple (x, y) or string: 'center', 'random', 'top-left', etc.")
        return cx, cy

    def _get_p_grid(self, dist):
        t = dist['type']
        min_p_global = dist.get('min_p', 0.0) 
        max_p_global = dist.get('max_p', 1.0)
        
        start_pos = dist.get('start_pos', (0, 0))
        end_pos = dist.get('end_pos', (1, 1))
        sx, sy = start_pos
        ex, ey = end_pos
        
        if not (0 <= sx < ex <= 1 and 0 <= sy < ey <= 1):
            raise ValueError(f"Invalid start/end positions: start={start_pos}, end={end_pos}. Must satisfy 0 <= start < end <= 1.")

        active_mask = (self.X >= sx) & (self.X <= ex) & (self.Y >= sy) & (self.Y <= ey)
        
        width = ex - sx
        height = ey - sy
        X_scaled = (self.X - sx) / width if width > 0 else np.full_like(self.X, 0.5)
        Y_scaled = (self.Y - sy) / height if height > 0 else np.full_like(self.Y, 0.5)
        
        p_grid = np.zeros_like(self.X)

        if t == 'linear':
            direction = dist.get('direction')
            if direction == 'horizontal-rightleft': p_grid = min_p_global + (max_p_global - min_p_global) * X_scaled
            elif direction == 'horizontal-leftright': p_grid = max_p_global - (max_p_global - min_p_global) * X_scaled
            elif direction == 'vertical-downup': p_grid = min_p_global + (max_p_global - min_p_global) * Y_scaled
            elif direction == 'vertical-updown': p_grid = max_p_global - (max_p_global - min_p_global) * Y_scaled
            else: raise ValueError("Direction must be 'horizontal-rightleft'/'horizontal-leftright' or 'vertical-downup'/'vertical-updown'")
        
        elif t == 'radial':
            cx, cy = self._get_position(dist)
            dist_grid = np.sqrt((X_scaled - cx)**2 + (Y_scaled - cy)**2)
            max_dist = np.max(dist_grid[active_mask]) if np.any(active_mask) else 1.0
            dist_norm = dist_grid / max_dist if max_dist > 0 else np.zeros_like(dist_grid)
            p_grid = max_p_global + (min_p_global - max_p_global) * dist_norm
        
        elif t == 'quadrant':
            setups = dist.get('quadrant_setups', [dist])
            for setup in setups:
                min_p, max_p = setup.get('min_p', min_p_global), setup.get('max_p', max_p_global)
                quadrant_str, peak, diffusion = setup.get('quadrant', 'top-right').lower(), setup.get('peak', 'center').lower(), setup.get('diffusion', 'linear').lower()
                if quadrant_str == 'top-left': quad_mask, corner_point = ((X_scaled <= 0.5) & (Y_scaled <= 0.5)), (0.0, 0.0)
                elif quadrant_str == 'top-right': quad_mask, corner_point = ((X_scaled > 0.5) & (Y_scaled <= 0.5)), (1.0, 0.0)
                elif quadrant_str == 'bottom-left': quad_mask, corner_point = ((X_scaled <= 0.5) & (Y_scaled > 0.5)), (0.0, 1.0)
                elif quadrant_str == 'bottom-right': quad_mask, corner_point = ((X_scaled > 0.5) & (Y_scaled > 0.5)), (1.0, 1.0)
                else: raise ValueError("Quadrant must be one of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.")
                center_point = (0.5, 0.5)
                if diffusion == 'linear':
                    v = (corner_point[0] - center_point[0], corner_point[1] - center_point[1])
                    v_dot_v = v[0]**2 + v[1]**2
                    if v_dot_v == 0: dist_norm = np.zeros_like(X_scaled)
                    else:
                        u_x, u_y = X_scaled - center_point[0], Y_scaled - center_point[1]
                        proj_grid = (u_x * v[0] + u_y * v[1]) / v_dot_v
                        dist_norm = np.clip(proj_grid, 0, 1)
                elif diffusion == 'radial':
                    dist_from_center = np.sqrt((X_scaled - center_point[0])**2 + (Y_scaled - center_point[1])**2)
                    max_dist_to_corner = np.sqrt((corner_point[0] - center_point[0])**2 + (corner_point[1] - center_point[1])**2)
                    if max_dist_to_corner == 0: dist_norm = np.zeros_like(dist_from_center)
                    else: dist_norm = np.clip(dist_from_center / max_dist_to_corner, 0, 1)
                else: raise ValueError("Diffusion for quadrant must be 'linear' or 'radial'.")
                if peak == 'center': quad_p_grid = max_p - (max_p - min_p) * dist_norm
                elif peak == 'corner': quad_p_grid = min_p + (max_p - min_p) * dist_norm
                else: raise ValueError("Peak for quadrant must be 'center' or 'corner'.")
                p_grid[quad_mask & active_mask] = quad_p_grid[quad_mask & active_mask]
        
        elif t == 'custom':
            custom_mask = dist.get('custom_mask')
            if custom_mask is not None and custom_mask.shape == (self.size, self.size):
                # Create a base radial gradient from the center, scaled from max_p down to min_p.
                cx, cy = 0.5, 0.5
                dist_from_center = np.sqrt((X_scaled - cx)**2 + (Y_scaled - cy)**2)
                max_dist = np.sqrt(0.5**2 + 0.5**2) 
                dist_norm = dist_from_center / max_dist if max_dist > 0 else np.zeros_like(dist_from_center)
                base_p_grid = max_p_global + (min_p_global - max_p_global) * dist_norm
                
                # Apply the mask. Multiplying by the mask zeros out any area the user hasn't drawn on.
                p_grid = base_p_grid * custom_mask
            else:
                p_grid = np.zeros_like(self.X)

        else:
            raise ValueError(f"Unknown distribution type: {t}")
        final_p_grid = np.where(active_mask, p_grid, 0)
        return np.clip(final_p_grid, 0, 1)


    def get_stimuli(self, coordinates, shape='square', radius=0, mode='count', weighting='uniform', sigma=None):
        """
        Get stimuli values at or around a given coordinate.

        Parameters:
        - coordinates: tuple (int, int), the (x, y) center coordinate.
        - shape: str (default 'square'), the shape of the area.
                 Options: 'circle', 'square'.
        - radius: int (default 0), the radius of the area to consider. If 0,
                  only the center coordinate is checked.
        - mode: str (default 'count'), the operation to perform.
                Options: 'concentration' (returns a float from 0.0 to 1.0),
                         'count' (returns an integer count of active stimuli).
        - weighting: str (default 'uniform'), the weighting method for stimuli
                     within the radius. Only used when mode is 'concentration'.
                     Options: 'uniform', 'linear', 'gaussian'.
        - sigma: float, optional. The standard deviation for 'gaussian'
                 weighting. Defaults to radius / 2.0.

        Returns:
        - list of int or list of float: Based on the mode, returns a list of
          stimuli values, counts, or concentrations for each layer.
        
        Raises:
        - IndexError: if the coordinates are outside the grid boundaries.
        - ValueError: if shape, weighting, or mode parameters are invalid.
        """
        x, y = coordinates
        
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise IndexError(f"Coordinates ({x}, {y}) are out of bounds for a grid of size {self.size}x{self.size}.")
            
        if radius == 0:
            return [int(layer[y, x]) for layer in self.layers]

        jj, ii = np.indices((self.size, self.size))

        if shape == 'circle':
            distances = np.sqrt((ii - x)**2 + (jj - y)**2)
        elif shape == 'square':
            distances = np.maximum(np.abs(ii - x), np.abs(jj - y))
        else:
            raise ValueError("Shape must be 'circle' or 'square'.")
            
        mask = distances <= radius

        if not np.any(mask):
            return [0.0 if mode == 'concentration' else 0] * self.num_layers

        if mode == 'count':
            counts = []
            for layer in self.layers:
                count = np.sum(layer[mask])
                counts.append(int(count))
            return counts

        elif mode == 'concentration':
            if weighting == 'uniform':
                weights = mask.astype(float)
            elif weighting == 'linear':
                weights = np.maximum(0, 1 - distances / radius)
                weights[~mask] = 0
            elif weighting == 'gaussian':
                if sigma is None:
                    sigma = radius / 2.0
                if sigma <= 0:
                    raise ValueError("Sigma for Gaussian weighting must be positive.")
                weights = np.exp(-distances**2 / (2 * sigma**2))
                weights[~mask] = 0
            else:
                raise ValueError("Weighting must be 'uniform', 'linear', or 'gaussian'.")

            total_weight = np.sum(weights)
            if total_weight == 0:
                return [0.0] * self.num_layers
                
            concentrations = []
            for layer in self.layers:
                weighted_sum = np.sum(layer * weights)
                concentration = weighted_sum / total_weight
                concentrations.append(concentration)
                
            return concentrations
        
        else:
            raise ValueError("Mode must be 'concentration' or 'count'.")

    def visualize(self, combined_only=False):
        """
        Visualize the petri dish layers, old version; new version is "visualize_layers" and "_combined".
        
        Parameters:
        - combined_only: bool, if True, returns only the combined RGB grid.
        """
        colors = [colorsys.hsv_to_rgb(i / self.num_layers, 1, 1) for i in range(self.num_layers)]
        
        # Create the combined RGB grid by blending colors based on intensity
        combined = np.ones((self.size, self.size, 3), dtype=float)
        color_sums = np.zeros((self.size, self.size, 3), dtype=float)
        intensity_sums = np.zeros((self.size, self.size), dtype=float)
        
        for i in range(self.num_layers):
            intensity = self.layers[i]
            color = np.array(colors[i])
            color_sums += intensity[:, :, np.newaxis] * color
            intensity_sums += intensity
        
        mask_active = intensity_sums > 0
        if np.any(mask_active):
            # Normalize the color sums by the total intensity at each pixel
            combined[mask_active] = color_sums[mask_active] / intensity_sums[mask_active][:, np.newaxis]
        
        if combined_only:
            return combined

        # Full visualization with subplots
        fig, axs = plt.subplots(1, self.num_layers + 1, figsize=(4 * (self.num_layers + 1), 4))
        for i in range(self.num_layers):
            color = colors[i]
            # Create a colormap for this layer from white to the layer's color
            cmap_colors = np.array([np.linspace(1, c, self.stimuli_intensity + 1) for c in color]).T
            layer_cmap = ListedColormap(cmap_colors)
            
            axs[i].imshow(self.layers[i], cmap=layer_cmap, vmin=0, vmax=self.stimuli_intensity)
            axs[i].set_title(f'Layer {i}')
            axs[i].axis('off')
        
        axs[-1].imshow(combined)
        axs[-1].set_title('Combined')
        axs[-1].axis('off')
        
        plt.tight_layout()
        plt.show()

    def visualize_layer(self, layer_index):
        """Generates a matplotlib figure for a single specified layer."""
        if not (0 <= layer_index < self.num_layers):
            return None
        
        color = colorsys.hsv_to_rgb(layer_index / self.num_layers, 1, 1)
        cmap_colors = np.array([np.linspace(1, c, self.stimuli_intensity + 1) for c in color]).T
        layer_cmap = ListedColormap(cmap_colors)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.layers[layer_index], cmap=layer_cmap, vmin=0, vmax=self.stimuli_intensity)
        ax.axis('off')
        
        return fig

    def visualize_combined(self):
        """
        Generates a matplotlib figure for the combined view of all layers using an
        additive color model.
        """
        if self.num_layers == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No layers to display.', ha='center', va='center')
            return fig

        # Get the base HSV colors for each layer
        colors = [colorsys.hsv_to_rgb(i / self.num_layers, 1, 1) for i in range(self.num_layers)]

        # Start with a black canvas for additive color blending
        combined_rgb = np.zeros((self.size, self.size, 3), dtype=float)

        # Handle the case where stimuli_intensity might be 0 to avoid division by zero
        max_intensity = float(self.stimuli_intensity) if self.stimuli_intensity > 0 else 1.0

        for i in range(self.num_layers):
            # Get the intensity grid for the current layer
            intensity = self.layers[i].astype(float)
            
            # Normalize the intensity to create a brightness/opacity map for this layer's color
            normalized_intensity = intensity / max_intensity
            
            # Get the color for this layer
            color = np.array(colors[i])
            
            # Add this layer's color, scaled by its intensity, to the combined image
            combined_rgb += normalized_intensity[:, :, np.newaxis] * color

        # Clip the values to the valid RGB range [0, 1] in case overlapping intensities sum > 1
        combined_rgb = np.clip(combined_rgb, 0, 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(combined_rgb)
        ax.axis('off')
        
        return fig

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="Petri Dish Environment Builder")
st.title("🔬 Petri Dish Environment Builder")
st.write("Interactively design and visualize stimuli distributions for your 2D environment.")

# --- Initialize Session State ---
if 'distributions' not in st.session_state:
    st.session_state.distributions = []

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Global Settings")
    grid_size = st.slider("Grid Size", 50, 500, 200, 10)
    stimuli_intensity = st.slider("Max Stimuli Intensity (n)", 1, 50, 9, 1)
    
    st.header("Layer Management")
    if st.button("＋ Add New Layer"):
        new_layer = {'type': 'linear', 'direction': 'horizontal-rightleft', 'min_p': 0.0, 'max_p': 1.0, 'start_pos': (0.0, 0.0), 'end_pos': (1.0, 1.0)}
        st.session_state.distributions.append(new_layer)
        
    st.header("Export Environment")
    st.write("Save the current setup list to a pickle file.")
    if st.session_state.distributions:
        export_dish = PetriDish(
            size=grid_size, 
            distributions=st.session_state.distributions, 
            stimuli_intensity=stimuli_intensity
        )
        # # Create a deep copy for serialization to avoid including the large 'custom_mask' array
        # export_distributions = []
        # for dist in st.session_state.distributions:
        #     dist_copy = dist.copy()
        #     if 'custom_mask' in dist_copy:
        #         del dist_copy['custom_mask'] # Remove mask before saving
        #     export_distributions.append(dist_copy)

    # if st.session_state.dish:
        env_bytes = pickle.dumps(export_dish)
        st.download_button(
            label="Download dish_env.pkl",
            data=env_bytes,
            file_name="dish_env.pkl",
            mime="application/octet-stream"
        )
    else:
        st.info("Add at least one layer to enable export.")


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
                st.subheader(f"Layer {i}")
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
        dish = PetriDish(size=grid_size, distributions=st.session_state.distributions, stimuli_intensity=stimuli_intensity)
        
        # --- Part 2: Individual Layer Visualizations ---
        st.header("Individual Layer Views")
        st.write("Each view below corresponds to the configuration column directly above it.")
        vis_cols = st.columns(len(st.session_state.distributions))
        for i, col in enumerate(vis_cols):
            with col:
                fig_layer = dish.visualize_layer(i)
                if fig_layer:
                    st.pyplot(fig_layer)

        # --- Part 3: Combined View ---
        
        # Create three columns to center the combined view
        st.write("##")
        _left_spacer, center_col, _right_spacer = st.columns([1, 1, 1])
        with center_col:
            st.header("Combined Environment View")
            fig_combined = dish.visualize_combined()
            st.pyplot(fig_combined)

    except ValueError as e:
        st.error(f"Error generating environment: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")