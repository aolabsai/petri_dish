# petridish class
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.colors import ListedColormap

# assay class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from archs.arch0 import updateArch, RADIUS,OUTPUT_SIZE
import random

import ao_core as ao
# RADIUS = 2
# --- PetriDish Class Definition ---
# The class is updated with a corrected visualize_combined method.

PLEASURE_LAYER_INDEX = 0
PAIN_LAYER_INDEX = 2
class PetriDish:
    def __init__(self, size=100, num_layers=3, distributions=None, stimuli_intensity=9):
        """
        Initialize the PetriDish simulation.

        Parameters:
        - size: int, the size of the square grid (size x size).
        - num_layers: int, the number of layers (each representing a different stimulus).
        - distributions: list of dicts, parameters for the distribution of each layer.
          Each dict can have:
            - 'type': 'linear', 'radial', 'quadrant', 'custom', or 'empty'.
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
            - For 'empty':
              - This type creates a blank layer with no stimuli.
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

        # --- Added block to handle the 'empty' distribution type ---
        elif t == 'empty':
            # p_grid is already initialized to zeros, so this creates an empty layer.
            pass

        else:
            raise ValueError(f"Unknown distribution type: {t}")
            
        final_p_grid = np.where(active_mask, p_grid, 0)
        return np.clip(final_p_grid, 0, 1)


    # def get_stimuli(self, coordinates, shape='square', radius=RADIUS, mode='count', weighting=None, sigma=None):
    
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
        # x, y = coordinates
        
        # if not (0 <= x < self.size and 0 <= y < self.size):
        #     raise IndexError(f"Coordinates ({x}, {y}) are out of bounds for a grid of size {self.size}x{self.size}.")
            
        # if radius == 0:
        #     return [int(layer[y, x]) for layer in self.layers]

        # jj, ii = np.indices((self.size, self.size))

        # if shape == 'circle':
        #     distances = np.sqrt((ii - x)**2 + (jj - y)**2)
        # elif shape == 'square':
        #     distances = np.maximum(np.abs(ii - x), np.abs(jj - y))
        # else:
        #     raise ValueError("Shape must be 'circle' or 'square'.")
            
        # mask = distances <= radius


        # if not np.any(mask):
        #     return [0.0 if mode == 'concentration' else 0] * self.num_layers

        # if mode == 'count':
        #     counts = []
        #     radius_layers = []
        #     for layer in self.layers:
        #         count = np.sum(layer[mask])
        #         radius_layers.append(layer[mask])
        #         counts.append(int(count))
        #     return counts, radius_layers

        # elif mode == 'concentration':
        #     if weighting == 'uniform':
        #         weights = mask.astype(float)
        #     elif weighting == 'linear':
        #         weights = np.maximum(0, 1 - distances / radius)
        #         weights[~mask] = 0
        #     elif weighting == 'gaussian':
        #         if sigma is None:
        #             sigma = radius / 2.0
        #         if sigma <= 0:
        #             raise ValueError("Sigma for Gaussian weighting must be positive.")
        #         weights = np.exp(-distances**2 / (2 * sigma**2))
        #         weights[~mask] = 0
        #     else:
        #         raise ValueError("Weighting must be 'uniform', 'linear', or 'gaussian'.")

        #     total_weight = np.sum(weights)
        #     if total_weight == 0:
        #         return [0.0] * self.num_layers
                
        #     concentrations = []
        #     for layer in self.layers:
        #         weighted_sum = np.sum(layer * weights)
        #         concentration = weighted_sum / total_weight
        #         concentrations.append(concentration)
                
        #     return concentrations
        
        # else:
        #     raise ValueError("Mode must be 'concentration' or 'count'.")
    def get_stimuli(self, coordinates, shape='square', radius=RADIUS, mode='count', weighting=None, sigma=None):
        """
        Get stimuli values at or around a given coordinate with consistent output size.
        Pads with zeros when the sensory area extends beyond grid boundaries.
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

        # Calculate the full sensory area bounds (may extend outside grid)
        if shape == 'square':
            min_x, max_x = x - radius, x + radius
            min_y, max_y = y - radius, y + radius
            expected_width = 2 * radius + 1
            expected_height = 2 * radius + 1
            expected_size = expected_width * expected_height
        elif shape == 'circle':
            min_x, max_x = x - radius, x + radius
            min_y, max_y = y - radius, y + radius
            expected_width = 2 * radius + 1
            expected_height = 2 * radius + 1
            # For circle, we'll create the full area first then mask
            expected_size = expected_width * expected_height
        else:
            raise ValueError("Shape must be 'circle' or 'square'.")

        # Calculate actual bounds within the grid
        actual_min_x = max(0, min_x)
        actual_max_x = min(self.size - 1, max_x)
        actual_min_y = max(0, min_y)
        actual_max_y = min(self.size - 1, max_y)

        # Create coordinate grids for the FULL sensory area (including out-of-bounds)
        full_ii, full_jj = np.meshgrid(
            np.arange(min_x, max_x + 1), 
            np.arange(min_y, max_y + 1),
            indexing='xy'
        )

        if shape == 'circle':
            distances = np.sqrt((full_ii - x)**2 + (full_jj - y)**2)
            full_mask = distances <= radius
        else:  # square
            full_mask = np.ones_like(full_ii, dtype=bool)

        # Create masks for what's inside vs outside the grid
        inside_grid_mask = (
            (full_ii >= 0) & (full_ii < self.size) & 
            (full_jj >= 0) & (full_jj < self.size)
        )
        
        # Combine with shape mask
        valid_mask = full_mask & inside_grid_mask
        
        # Debug information
        total_cells = np.sum(full_mask)
        valid_cells = np.sum(valid_mask)
        padded_cells = total_cells - valid_cells
        
        # print(f"Position ({x}, {y}): Total={total_cells}, Valid={valid_cells}, Padded={padded_cells}")
        
        if not np.any(valid_mask):
            # No valid cells (shouldn't happen with reasonable radius)
            if mode == 'count':
                padded_layers = [np.zeros(total_cells, dtype=int) for _ in self.layers]
                return [0] * self.num_layers, padded_layers
            else:
                return [0.0] * self.num_layers

        if mode == 'count':
            counts = []
            padded_layers = []
            
            for layer in self.layers:
                # Create padded layer data
                padded_layer_data = np.zeros(full_mask.shape, dtype=layer.dtype)
                
                # Fill in the valid (in-bounds) data
                valid_coords = np.where(valid_mask)
                grid_coords_i = full_ii[valid_coords]
                grid_coords_j = full_jj[valid_coords]
                
                padded_layer_data[valid_coords] = layer[grid_coords_j, grid_coords_i]
                
                # Extract only the cells within the shape mask
                shape_layer_data = padded_layer_data[full_mask]
                
                count = np.sum(shape_layer_data)
                padded_layers.append(shape_layer_data)
                counts.append(int(count))
                
            return counts, padded_layers

        elif mode == 'concentration':
            if weighting == 'uniform':
                weights = full_mask.astype(float)
            elif weighting == 'linear':
                if shape == 'circle':
                    distances = np.sqrt((full_ii - x)**2 + (full_jj - y)**2)
                else:
                    distances = np.maximum(np.abs(full_ii - x), np.abs(full_jj - y))
                weights = np.maximum(0, 1 - distances / radius)
                weights[~full_mask] = 0
            elif weighting == 'gaussian':
                if sigma is None:
                    sigma = radius / 2.0
                if sigma <= 0:
                    raise ValueError("Sigma for Gaussian weighting must be positive.")
                if shape == 'circle':
                    distances = np.sqrt((full_ii - x)**2 + (full_jj - y)**2)
                else:
                    distances = np.maximum(np.abs(full_ii - x), np.abs(full_jj - y))
                weights = np.exp(-distances**2 / (2 * sigma**2))
                weights[~full_mask] = 0
            else:
                raise ValueError("Weighting must be 'uniform', 'linear', or 'gaussian'.")

            # Only apply weights to valid (in-bounds) cells
            weights[~inside_grid_mask] = 0
            
            total_weight = np.sum(weights)
            if total_weight == 0:
                return [0.0] * self.num_layers
                
            concentrations = []
            for layer in self.layers:
                # Create padded layer (out-of-bounds areas remain 0)
                padded_layer_data = np.zeros(full_mask.shape, dtype=float)
                
                valid_coords = np.where(valid_mask)
                grid_coords_i = full_ii[valid_coords]
                grid_coords_j = full_jj[valid_coords]
                
                padded_layer_data[valid_coords] = layer[grid_coords_j, grid_coords_i]
                
                weighted_sum = np.sum(padded_layer_data * weights)
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
    




class Assay:
    """
    Manages and runs experiments with multiple agents on a PetriDish.
    Includes a debug mode if agent_archs="random".
    """
    def __init__(self, petri_dish, num_agents=5, start_logic='random', start_positions=None, agent_archs="random", assay_loadagents="", save_agent_meta=False, steps=1000):
        self.metainfo = 7
        self.dish = petri_dish
        
        self.debug_mode = (agent_archs == "random")
        self.random = self.debug_mode
        
        if type(assay_loadagents) is Assay:
            print("-----AGENTS LOADED FROM SUPPLIED ASSAY-----")
            self.num_agents = assay_loadagents.num_agents
            self.agent_archs = assay_loadagents.agent_archs
            self.debug_mode = getattr(assay_loadagents, 'debug_mode', False)
        else:
            self.num_agents = num_agents
            self.agent_archs = agent_archs

        if self.debug_mode:
            self.num_learning_types = 0
            print("-----ASSAY IN RANDOM/DEBUG MODE-----")
        # else:
        #     self.num_learning_types = len(self.agent_archs.arch_c)

        self.history = np.zeros((steps, self.num_agents, 2), dtype=int)
        self.meta_history = np.zeros((steps, self.num_agents, self.metainfo), dtype=object)
        self.save_agent_meta = save_agent_meta
         
        self._initialize_agents(start_logic, start_positions, assay_loadagents)
        self.step = 0

        self.INSTINCTS = False
        self.sensory_shape = "square"
        self.alpha = 0.2   #let's call this learning_rate
        self.gamma = 0.95 #discount_factor for long_term planning
        self.epsilon = 0.2  #exploration_rate
        self.sensory_mode = "count"
        self.sensory_radius = RADIUS  
        self.sensory_area_size = (2 * RADIUS + 1) ** 2  # (2*2 + 1)^2 = 5^2 = 25 cells
        
        # Each cell has 3 layers, each layer gets binary encoded
        self.sensory_binary_neurons = self.sensory_area_size * self.dish.stimuli_intensity

        self.ACTIONS = ['move_forward', 'move_right', 'move_left', 'move_backward']
        self.ACTIONS_BIN = {"move_forward": [0, 0], "move_right": [0, 1], "move_left": [1, 0], "move_backward": [1, 1]}
        self.HEADINGS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.HEADINGS_BIN = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}


    def _initialize_agents(self, start_logic, start_positions, assay_loadagents):
        if start_positions:
            positions = start_positions
        else:
            s = self.dish.size
            mid, q = s // 2, s // 4
            logic_map = {
                'random': [tuple(np.random.randint(0, s, 2)) for _ in range(self.num_agents)],
                'center': [(mid, mid)] * self.num_agents,
                'cardinal': [(mid, 0), (mid, s-1), (0, mid), (s-1, mid)],
                'quadrants': [(q, q), (q, s-q), (s-q, q), (s-q, s-q)],
                'corners': [(0, 0), (0, s-1), (s-1, 0), (s-1, s-1)]
            }
            if start_logic not in logic_map:
                raise ValueError(f"Unknown start_logic: '{start_logic}'.")
            positions = logic_map[start_logic]
        
        self.agents = np.empty(self.num_agents, dtype=object)
        for i, pos in enumerate(positions[:self.num_agents]):
            if not (0 <= pos[0] < self.dish.size and 0 <= pos[1] < self.dish.size):
                raise ValueError(f"Start position {pos} is out of bounds.")

            if type(assay_loadagents) is Assay:
                agent = assay_loadagents.agents[i]['agent']
            else:
                if self.debug_mode:
                    agent = None
                else:
                    agent = ao.Agent(self.agent_archs, save_meta=self.save_agent_meta)
                    # for c in self.agent_archs.C_types_names:
                    #     setattr(agent, c, 0)

            self.agents[i] = {
                'id': i,
                'pos': pos,
                'heading': np.random.randint(0, 4),
                'agent': agent
            }
            self.history[0, i, 0] = pos[0]
            self.history[0, i, 1] = pos[1]
        
    def _convert_stimuli_to_binary(self, radius_layers):

        agent_input_binary = np.array([[layer[i] for layer in radius_layers] for i in range(len(radius_layers[0]))], dtype=np.uint8).reshape(-1)
        # print(agent_input_binary.shape)
        # agent_input_binary = []
        # for val in stimuli_input:
        #     val_capped = min(int(round(val)), self.sensory_binary_neurons)
        #     input_binary = np.zeros(self.sensory_binary_neurons)
        #     if val_capped > 0:
        #         input_binary[0:val_capped] = 1
        #     agent_input_binary.extend(input_binary)
        # agent_input_binary = np.array(agent_input_binary)
        return agent_input_binary

    # def _get_agent_action(self, agent):
    #     if self.random:
    #         return np.random.choice(self.ACTIONS)
        
    #     stimuli_input = self.dish.get_stimuli(agent['pos'], shape=self.sensory_shape, radius=self.sensory_radius,  mode=self.sensory_mode)
    #     agent_input_binary = self._convert_stimuli_to_binary(agent_input_binary)

    #     agent_action_binary = agent['agent'].next_state(agent_input_binary, INSTINCTS=self.INSTINCTS, print_result=False)
        
    #     if np.array_equal(agent_action_binary, [1]):
    #         return self.ACTIONS[0]
    #     else:
    #         return np.random.choice(self.ACTIONS[1:])
            
    def set_agent_hyperparameters(self, C_impression_initial=10, C_impression_match=1, C_pruning=1, C_pruning_cutoff=5, save_C_info=True,
                                  saturation_threshold=10, refractory_period=5, saturation_refractory=True):
        if self.debug_mode:
            print("Cannot set hyperparameters in random/debug mode.")
            return

        for agent_dict in self.agents:
            a = agent_dict['agent']
            # a._save_C_info = save_C_info
            # a.C_impression_initial = C_impression_initial
            # a.C_impression_match = C_impression_match
            # a.C_pruning = C_pruning
            # a.C_pruning_cutoff = C_pruning_cutoff

            a._saturation_refractory = saturation_refractory
            a.saturation_threshold = saturation_threshold
            a.refractory_period = refractory_period



    def get_agent_action(self, agent_dict,  exploration_rate = None):
        agent = agent_dict['agent']
        if exploration_rate is None:
            exploration_rate = self.epsilon

        if random.random() < exploration_rate:  # e.g., 10% of the time
            # For random actions, we still need to create the binary input
            stimuli_input,radius_layers = self.dish.get_stimuli(
                agent_dict['pos'], 
                shape=self.sensory_shape, 
                radius=self.sensory_radius, 
                mode=self.sensory_mode
            )
            agent_input_binary = self._convert_stimuli_to_binary(radius_layers)
            
            
            agent_input_binary = np.append(agent_input_binary, self.HEADINGS_BIN[agent_dict['heading']])
            
            random_action = random.choice(self.ACTIONS)
            
            # Append random action to binary input
            agent_input_binary = np.append(agent_input_binary, self.ACTIONS_BIN[random_action])
            # print(agent_input_binary.shape)
            q_value = sum(agent_dict['agent'].next_state(agent_input_binary, unsequenced=False))
            # action_q_values_prediction_dict[action] = q_value
            max_q_action = (random_action, q_value)  #max(action_q_values_prediction_dict.items(), key=lambda item: item[1])
            
            return agent_input_binary, max_q_action  # Return both
            
        # Get sensory input at this position
        stimuli_input, radius_layers = self.dish.get_stimuli(
            agent_dict['pos'], 
            shape=self.sensory_shape, 
            radius=self.sensory_radius, 
            mode=self.sensory_mode
        )
        action_q_values_prediction_dict = {}
        
        heading = agent_dict['heading']
        for action in self.ACTIONS:
            agent_action = self.ACTIONS_BIN[action]
            # Convert to binary input for WNN
            agent_input_binary = self._convert_stimuli_to_binary(radius_layers)
            agent_input_binary = np.append(agent_input_binary, self.HEADINGS_BIN[heading])
            agent_input_binary = np.append(agent_input_binary, agent_action)
            # print(agent_input_binary.shape)
            q_value = sum(agent_dict['agent'].next_state(agent_input_binary, unsequenced=False))
            action_q_values_prediction_dict[action] = q_value
        max_q_action = max(action_q_values_prediction_dict.items(), key=lambda item: item[1])
        # max_q_action = max(action_q_values_prediction_dict, key=action_q_values_prediction_dict.get)
        return agent_input_binary, max_q_action


    # def _convert_stimuli_to_binary(self, stimuli_input):
    #     agent_input_binary = []
    #     for val in stimuli_input:
    #         val_capped = min(int(round(val)), self.sensory_binary_neurons)
    #         input_binary = np.zeros(self.sensory_binary_neurons)
    #         if val_capped > 0:
    #             input_binary[0:val_capped] = 1
    #         agent_input_binary.extend(input_binary)
    #     agent_input_binary = np.array(agent_input_binary)
    #     return agent_input_binary

    def calculate_reward(self,stimuli_values):
        """
        Calculate reward based on biological interpretation of layers:
        Layer 0: Food (positive resource)
        Layer 1: Neutral chemical (no direct reward impact)  
        Layer 2: Toxin (negative resource)
        """
        food_level = stimuli_values[0]
        neutral_level = stimuli_values[1] 
        toxin_level = stimuli_values[2]
        
        # Base reward components
        food_reward = food_level * 5.0  # Food is good
        toxin_penalty = toxin_level * -1.5  # Toxins are bad
        # movement_cost = -0.1  # Small cost for moving (encourages efficiency)
        
        # Interaction effects
        if food_level > 0 and toxin_level > 0:
            contamination_penalty = -0.5 * min(food_level, toxin_level)
        else:
            contamination_penalty = 0.0
        
        total_reward = food_reward + toxin_penalty + contamination_penalty #+ movement_cost
        
        return total_reward
            
    def update_agent_position_and_heading(self, agent_dict, action):
        
        x, y = agent_dict['pos']
        if action == 'move_forward':
            dy, dx = self.HEADINGS[agent_dict['heading']]
            new_x, new_y = x + (dx*RADIUS), y + (dy*RADIUS)

        elif action == 'move_backward':
            agent_dict['heading'] = (agent_dict['heading'] + 2) % 4
            dy, dx = self.HEADINGS[agent_dict['heading']] 
            new_x, new_y = x + (dx*RADIUS), y + (dy*RADIUS)
        elif action == 'move_right':
            agent_dict['heading'] = (agent_dict['heading'] + 1) % 4
            dy, dx = self.HEADINGS[agent_dict['heading']]
            new_x, new_y = x + (dx*RADIUS), y + (dy*RADIUS)
        elif action == 'move_left':
            agent_dict['heading'] = (agent_dict['heading'] - 1 + 4) % 4
            dy, dx = self.HEADINGS[agent_dict['heading']]
            new_x, new_y = x + (dx*RADIUS), y + (dy*RADIUS)
        if 0 <= new_x < self.dish.size and 0 <= new_y < self.dish.size:
            agent_dict['pos'] = (new_x, new_y)
        # return agent_dict

    def update_assay_agents(self,agent_dict, agent_input, new_q):
        agent_input = np.array(agent_input, dtype=np.int8)
        clipped_q = int(round(np.clip(new_q, 0, OUTPUT_SIZE)))
        new_q_label = np.zeros(OUTPUT_SIZE, dtype=np.int8)
        new_q_idx = np.random.randint(0,OUTPUT_SIZE, size = (clipped_q))
        new_q_label[new_q_idx] = 1
        agent_dict['agent'].next_state_batch(agent_input.reshape(1,-1), new_q_label.reshape(1,-1))
    
    def pretrain_random(self, steps):
        """Pretrain agents by systematically exploring all possible state-action combinations"""
        if self.debug_mode:
            print("Cannot pretrain agents in random/debug mode.")
            return
            
        for i, agent_dict in enumerate(self.agents):
            agent_dict['agent'].reset_state()
            agent = agent_dict['agent']
            
            original_pos = agent_dict['pos']
            original_heading = agent_dict['heading']
            
            # Systematic exploration strategy
            for step in range(steps):
                # Move agent to different positions systematically
                if step < steps // 2:
                    # First half: grid sweep
                    grid_x = (step * 7) % self.dish.size  # Prime number for better coverage
                    grid_y = (step * 11) % self.dish.size
                    agent_dict['pos'] = (grid_x, grid_y)
                else:
                    # Second half: random positions
                    agent_dict['pos'] = (
                        np.random.randint(0, self.dish.size),
                        np.random.randint(0, self.dish.size)
                    )
                
                # Try different headings
                agent_dict['heading'] = step % 4
            
                # Get sensory input at this position
                old_stimuli,radius_layers = self.dish.get_stimuli(
                    agent_dict['pos'], 
                    shape=self.sensory_shape, 
                    radius=self.sensory_radius, 
                    mode=self.sensory_mode
                )
                old_reward = self.calculate_reward(old_stimuli)
                agent_input_binary, action_value = self.get_agent_action(agent_dict, exploration_rate =0.5)
                agent_action, q_value = action_value
                self.update_agent_position_and_heading(agent_dict,agent_action)
                new_stimuli,radius_layers = self.dish.get_stimuli(
                    agent_dict['pos'], 
                    shape=self.sensory_shape, 
                    radius=self.sensory_radius, 
                    mode=self.sensory_mode
                )
                new_reward = self.calculate_reward(new_stimuli)


                next_agent_input_binary, next_q_action_value = self.get_agent_action(agent_dict)
                next_agent_action, next_q_value = next_q_action_value
                # Get sensory input at this position

                # Combined reward strategies:
                # Option 1: Transition reward (difference)
                reward_received = new_reward - old_reward
                new_q = q_value + self.alpha * (reward_received + self.gamma * (next_q_value - q_value))
                self.update_assay_agents(agent_dict, agent_input_binary, new_q)
                
            # Restore original position and heading
            agent_dict['pos'] = original_pos
            agent_dict['heading'] = original_heading

    # def pretrain_random(self, steps):
    #     if self.debug_mode:
    #         print("Cannot pretrain agents in random/debug mode.")
    #         return
    #     for i, agent_dict in enumerate(self.agents):
    #         for s in range(steps):
    #             agent_dict['agent'].reset_state(training=True)
    #             agent_dict['agent'].reset_state()

    # def run_step(self, steps):
    #     for s in range(1, steps):
    #         if s >= self.history.shape[0]:
    #             print("Warning: History limit reached, assay memory full.")
    #             break
    #         self.step = s

    #         for i, agent_dict in enumerate(self.agents):
    #             action = self._get_agent_action(agent_dict)
    #             x, y = agent_dict['pos']
    #             if action == 'move_forward':
    #                 dy, dx = self.HEADINGS[agent_dict['heading']]
    #                 new_x, new_y = x + (dx*RADIUS), y + (dy*RADIUS)

    #             elif action == 'move_backward':
    #                 agent_dict['heading'] = (agent_dict['heading'])*-1
    #                 dy, dx = self.HEADINGS[heading] 
    #                 new_x, new_y = x - (dx*RADIUS), y - (dy*RADIUS)
    #             elif action == 'move_right':
    #                 agent_dict['heading'] = (agent_dict['heading'] + 1) % 4
    #                 dy, dx = self.HEADINGS[right_heading]
    #                 new_x, new_y = x + (dx*RADIUS), y + (dy*RADIUS)
    #             elif action == 'move_left':
    #                 agent_dict['heading'] = (agent_dict['heading'] - 1 + 4) % 4
    #                 dy, dx = self.HEADINGS[right_heading]
    #                 new_x, new_y = x + (dx*RADIUS), y + (dy*RADIUS)
    #             if 0 <= new_x < self.dish.size and 0 <= new_y < self.dish.size:
    #                 agent_dict['pos'] = (new_x, new_y)

    #             self.history[self.step, i, 0] = agent_dict['pos'][0]
    #             self.history[self.step, i, 1] = agent_dict['pos'][1]
                
    #             if self.debug_mode:
    #                 self.meta_history[self.step, i, :self.metainfo] = np.random.rand(self.metainfo) * 10
    #                 # if self.num_learning_types > 0:
    #                 #     self.meta_history[self.step, i, self.metainfo:] = np.random.randint(0, s, self.num_learning_types)
    #             else:
    #                 agent_instance = agent_dict['agent']
                    
    #                 self.meta_history[self.step, i, 0] = sum(agent_instance.astate[0, agent_instance.arch.I[0]])
    #                 self.meta_history[self.step, i, 1] = sum(agent_instance.astate[0, agent_instance.arch.I[1]])
    #                 self.meta_history[self.step, i, 2] = sum(agent_instance.astate[0, agent_instance.arch.I[2]])
    #                 self.meta_history[self.step, i, 3] = sum(agent_instance.astate[0, agent_instance.arch.Q[0]])
    #                 self.meta_history[self.step, i, 4] = sum(agent_instance.astate[0, agent_instance.arch.Q[1]])
    #                 self.meta_history[self.step, i, 5] = sum(agent_instance.astate[0, agent_instance.arch.Q[2]])
    #                 try: self.meta_history[self.step, i, 6] = agent_instance.neurons[agent_instance.arch.Z__flat[0]].outputs.size
    #                 except (AttributeError, IndexError): self.meta_history[self.step, i, 6] = self.meta_history[self.step-1, i, 6] if self.step > 0 else 0
    #                 # for c in range(self.num_learning_types):
    #                 #     self.meta_history[self.step, i, 7+c] = getattr(agent_instance, self.agent_archs.C_types_names[c])


    def run_step(self, steps):
        for s in range(1, steps):
            # if s >= self.history.shape[0]:
            #     print("Warning: History limit reached, assay memory full.")
            #     break
            self.step = s

            for i, agent_dict in enumerate(self.agents):
                # Get sensory input at this position
                stimuli_input, radius_layers = self.dish.get_stimuli(
                    agent_dict['pos'], 
                    shape=self.sensory_shape, 
                    radius=self.sensory_radius, 
                    mode=self.sensory_mode
                )

                agent_input_binary, action_value = self.get_agent_action(agent_dict)
                agent_action, q_value = action_value
                self.update_agent_position_and_heading(agent_dict,agent_action)
                next_agent_input_binary, next_q_action_value = self.get_agent_action(agent_dict)
                next_agent_action, next_q_value = next_q_action_value

                # Get sensory input at this position

                reward_received = self.calculate_reward(stimuli_input)
                new_q = q_value + self.alpha * (reward_received + self.gamma * next_q_value - q_value)
                self.update_assay_agents(agent_dict, agent_input_binary, new_q)

                self.history[self.step, i, 0] = agent_dict['pos'][0]
                self.history[self.step, i, 1] = agent_dict['pos'][1]
                
                if self.debug_mode:
                    self.meta_history[self.step, i, :self.metainfo] = np.random.rand(self.metainfo) * 10
                    # if self.num_learning_types > 0:
                    #     self.meta_history[self.step, i, self.metainfo:] = np.random.randint(0, s, self.num_learning_types)
                else:
                    agent_instance = agent_dict['agent']
                    
                    self.meta_history[self.step, i, 0] = sum(agent_instance.astate[0, agent_instance.arch.I[0]])
                    self.meta_history[self.step, i, 1] = sum(agent_instance.astate[0, agent_instance.arch.I[1]])
                    self.meta_history[self.step, i, 2] = sum(agent_instance.astate[0, agent_instance.arch.I[2]])
                    self.meta_history[self.step, i, 3] = sum(agent_instance.astate[0, agent_instance.arch.Q[0]])
                    self.meta_history[self.step, i, 4] = sum(agent_instance.astate[0, agent_instance.arch.Q[1]])
                    self.meta_history[self.step, i, 5] = sum(agent_instance.astate[0, agent_instance.arch.Q[2]])
                    try: self.meta_history[self.step, i, 6] = agent_instance.neurons[agent_instance.arch.Z__flat[0]].outputs.size
                    except (AttributeError, IndexError): self.meta_history[self.step, i, 6] = self.meta_history[self.step-1, i, 6] if self.step > 0 else 0
                    # for c in range(self.num_learning_types):
                    #     self.meta_history[self.step, i, 7+c] = getattr(agent_instance, self.agent_archs.C_types_names[c])
            if s >= self.history.shape[0]:
                print("Warning: History limit reached, assay memory full.")
                break

    def visualize(self, agents_to_show=None, interval=200, show_paths=True, mode='window'):
        # Old function, won't work with the new streamlit dashboard 
        fig, ax = plt.subplots(figsize=(8, 8))
        background = self.dish.visualize(combined_only=True)
        ax.imshow(background)
        ax.axis('off')

        if agents_to_show is None:
            agent_ids = list(range(self.num_agents))
        else:   
            if any(i < 0 or i >= self.num_agents for i in agents_to_show):
                raise ValueError("An invalid agent ID was provided in agents_to_show.")
            agent_ids = agents_to_show

        if not agent_ids:
            plt.close(fig)
            return

        full_agent_colors = plt.cm.jet(np.linspace(0, 1, self.num_agents))
        paths_to_show = self.history[:self.step + 1, agent_ids, :]
        
        path_lines = [ax.plot([], [], color=full_agent_colors[i], linewidth=1.5, alpha=0.7)[0] for i in agent_ids]
        agent_scatter = ax.scatter([], [], s=100, edgecolors='black', c=[])
        title = ax.set_title('')

        def init():
            for line in path_lines:
                line.set_data([], [])
            agent_scatter.set_offsets(np.empty((0, 2)))
            title.set_text('')
            return path_lines + [agent_scatter, title]

        def update(frame):
            if show_paths:
                for idx, line in enumerate(path_lines):
                    # path_data is now (frame, agent_idx, (x,y))
                    path_data = paths_to_show[:frame + 1, idx, :]
                    line.set_data(path_data[:, 0], path_data[:, 1])
            
            current_positions = paths_to_show[frame, :, :]
            agent_scatter.set_offsets(current_positions)
            agent_scatter.set_color([full_agent_colors[i] for i in agent_ids])
            title.set_text(f'Assay State at Step {frame}')
            
            return path_lines + [agent_scatter, title]

        anim = FuncAnimation(fig, update, frames=self.step + 1, init_func=init,
                             interval=interval, blit=True)
        
        if mode == 'inline':
            plt.close(fig)
            display(HTML(anim.to_jshtml()))
        elif mode == 'window':
            plt.show()
        else:
            plt.close(fig)
            raise ValueError("Invalid mode specified. Choose 'window' or 'inline'.")


    def export_data(self, to_step=None):
        if to_step is None: 
            to_step = self.step + 1
        
        agent_names = []
        agent_dfs = []
        for a in range(self.num_agents):
            agent_names.append(f"Agent_{a}")
            df = pd.DataFrame({
                'Time': np.arange(to_step),
                'stimuli0': self.meta_history[:to_step, a, 0],
                'stimuli1': self.meta_history[:to_step, a, 1],
                'stimuli2': self.meta_history[:to_step, a, 2],
                'neuronal0': self.meta_history[:to_step, a, 3],
                'neuronal1': self.meta_history[:to_step, a, 4],
                'neuronal2': self.meta_history[:to_step, a, 5],
                'Experience States': self.meta_history[:to_step, a, 6],
                'pos_x': self.history[:to_step, a, 0],
                'pos_y': self.history[:to_step, a, 1]
            })
            # if not self.debug_mode:
            #     for c in range(self.num_learning_types):
            #         df[self.agent_archs.C_types_names[c]] = self.meta_history[:to_step, a, 7+c]
            agent_dfs.append(df)
        
        self.agent_data = [ agent_names, agent_dfs ]
        self.env_data = self.dish.layers

        return True