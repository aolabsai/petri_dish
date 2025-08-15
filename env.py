
# code below generated with Grok 4 (thinking) and polished up further with Google Gemini
# https://grok.com/chat/cd16182e-2314-496f-a560-cf165dc6665d
# https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221u-B0YbunS-OQkpvWC1nkH69gzXhusZPW%22%5D,%22action%22:%22open%22,%22userId%22:%22102063868279130831651%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing



import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

import ao_core as ao


class PetriDish:
    def __init__(self, size=100, num_layers=3, distributions=None):
        """
        Initialize the PetriDish simulation.

        Parameters:
        - size: int, the size of the square grid (size x size).
        - num_layers: int, the number of layers (each representing a different stimulus).
        - distributions: list of dicts, parameters for the distribution of each layer.
          Each dict can have:
            - 'type': 'linear' or 'radial'
            - For 'linear':
              - 'direction': 'horizontal' or 'vertical'
              - 'min_p': float (default 0.0), minimum probability
              - 'max_p': float (default 1.0), maximum probability
            - For 'radial':
              - 'position': str or tuple (float, float), optional. Determines the location of the highest concentration point.
                - If tuple: specific coordinate (x, y) in [0,1]
                - If str: 'center' (default), 'random', 'top-left', 'top-right', 'bottom-left', 'bottom-right'
              - 'center': tuple (float, float), optional (backward compatibility, use 'position' instead)
              - 'min_p': float (default 0.0), probability at maximum distance
              - 'max_p': float (default 1.0), probability at the highest point

          If None, defaults to three example distributions.
        """
        if distributions is None:
            distributions = [
                {'type': 'linear', 'direction': 'horizontal', 'min_p': 0.0, 'max_p': 0.5},
                {'type': 'linear', 'direction': 'vertical', 'min_p': 0.0, 'max_p': 0.5},
                {'type': 'radial', 'center': (0.5, 0.5), 'min_p': 0.0, 'max_p': 0.5}
            ][:num_layers]

        self.size = size
        self.num_layers = num_layers
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        self.X, self.Y = np.meshgrid(x, y)
        self.layers = []
        for d in distributions[0:num_layers]:
            p_grid = self._get_p_grid(d)
            grid = np.random.binomial(1, p_grid).astype(int)
            self.layers.append(grid)

    def _get_position(self, dist):
        if 'position' not in dist:
            return dist.get('center', (0.5, 0.5))
        
        pos = dist['position']
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            cx, cy = map(float, pos)
        elif isinstance(pos, str):
            pos_lower = pos.lower()
            if pos_lower == 'center':
                cx, cy = 0.5, 0.5
            elif pos_lower == 'random':
                cx, cy = np.random.uniform(0, 1), np.random.uniform(0, 1)
            elif pos_lower == 'top-left':
                cx, cy = 0.0, 0.0
            elif pos_lower == 'top-right':
                cx, cy = 1.0, 0.0
            elif pos_lower == 'bottom-left':
                cx, cy = 0.0, 1.0
            elif pos_lower == 'bottom-right':
                cx, cy = 1.0, 1.0
            else:
                raise ValueError(f"Unknown position string: {pos}")
        else:
            raise ValueError("position must be tuple (x, y) or string: 'center', 'random', 'top-left', etc.")
        
        return cx, cy

    def _get_p_grid(self, dist):
        t = dist['type']
        min_p = dist.get('min_p', 0.0) # not sure why there is a ,0 and ,1 here in min and max_p
        max_p = dist.get('max_p', 1.0)
        if t == 'linear':
            direction = dist.get('direction')
            if direction == 'horizontal-rightleft':
                return min_p + (max_p - min_p) * self.X
            if direction == 'horizontal-leftright':
                return max_p - (max_p - min_p) * self.X
            if direction == 'vertical-downup':
                return min_p + (max_p - min_p) * self.Y
            elif direction == 'vertical-updown':
                return max_p - (max_p - min_p) * self.Y
            else:
                raise ValueError("Direction must be 'horizontal-rightleft' / 'horizontal-leftright'  or 'vertical-downup' / 'vertical-updown'")
        elif t == 'radial':
            cx, cy = self._get_position(dist)
            dist_grid = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            max_dist = np.max(dist_grid)
            dist_norm = dist_grid / max_dist if max_dist > 0 else np.zeros_like(dist_grid)
            return max_p + (min_p - max_p) * dist_norm
        else:
            raise ValueError(f"Unknown distribution type: {t}")

    def get_stimuli(self, coordinates, radius=0, shape='circle', weighting='uniform', sigma=None, mode='concentration'):
        """
        Get stimuli values at or around a given coordinate.

        Parameters:
        - coordinates: tuple (int, int), the (x, y) center coordinate.
        - radius: int (default 0), the radius of the area to consider. If 0,
                  only the center coordinate is checked.
        - shape: str (default 'circle'), the shape of the area.
                 Options: 'circle', 'square'.
        - weighting: str (default 'uniform'), the weighting method for stimuli
                     within the radius. Only used when mode is 'concentration'.
                     Options: 'uniform', 'linear', 'gaussian'.
        - sigma: float, optional. The standard deviation for 'gaussian'
                 weighting. Defaults to radius / 2.0.
        - mode: str (default 'concentration'), the operation to perform.
                Options: 'concentration' (returns a float from 0.0 to 1.0),
                         'count' (returns an integer count of active stimuli).

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
        Visualize the petri dish layers.
        
        Parameters:
        - combined_only: bool, if True, returns only the combined RGB grid.
        """
        # Create the combined RGB grid
        combined = np.ones((self.size, self.size, 3), dtype=float)
        color_sums = np.zeros((self.size, self.size, 3), dtype=float)
        counts = np.zeros((self.size, self.size), dtype=float)
        colors = [colorsys.hsv_to_rgb(i / self.num_layers, 1, 1) for i in range(self.num_layers)]
        
        for i in range(self.num_layers):
            mask = self.layers[i] == 1
            color_sums[mask] += colors[i]
            counts[mask] += 1
        
        mask_active = counts > 0
        if np.any(mask_active):
            combined[mask_active] = color_sums[mask_active] / counts[mask_active][:, None]
        
        if combined_only:
            return combined

        # Full visualization with subplots
        fig, axs = plt.subplots(1, self.num_layers + 1, figsize=(4 * (self.num_layers + 1), 4))
        for i in range(self.num_layers):
            layer_cmap = ListedColormap([[1, 1, 1], colors[i]])
            axs[i].imshow(self.layers[i], cmap=layer_cmap)
            axs[i].set_title(f'Layer {i}')
            axs[i].axis('off')
        
        axs[-1].imshow(combined)
        axs[-1].set_title('Combined')
        axs[-1].axis('off')
        
        plt.tight_layout()
        plt.show()


class Assay:
    """
    Manages and runs experiments with multiple agents on a PetriDish.
    """
    def __init__(self, petri_dish, num_agents=5, start_logic='random', start_positions=None, agent_archs="unit clam"):
        self.dish = petri_dish
        self.agents = []
        self.history = []
        self._initialize_agents(num_agents, start_logic, start_positions, agent_archs)

        # xx, yy = np.indices((self.dish.size, self.dish.size))
        # self.layers_proprioceptive = np.copy(self.dish.layers)
        
        # for x in np.nditer(xx):
        #     for y in np.nditer(yy):
        #         self.layers_proprioceptive[:, x, y] = self.dish.get_stimuli((x, y), radius=self.sensory_radius, shape=self.sensory_shape, mode=self.sensory_mode)
                
    ACTIONS = ['move_forward', 'turn_right', 'turn_left']
    HEADINGS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)} # N, E, S, W in (y,x)

    random = True # set to False to use AO agents
    INSTINCTS = False # set to True to trigger instinctual learning in AO agents
    sensory_binary_neurons = 10
    sensory_radius = 3
    sensory_shape = "circle"
    sensory_mode = "count"


    # def visualize_proprioceptive(self, combined_only=False):
    #     """
    #     Visualize the petri dish layers from the perspective of an agent, given the same parameters set for the assay. Helpful to visualize the fairness of the env for the learning agent.
        
    #     """
    #     colors = [colorsys.hsv_to_rgb(i / self.dish.num_layers, 1, 1) for i in range(self.dish.num_layers)]
        
    #     fig, axs = plt.subplots(1, self.dish.num_layers, figsize=(4 * (self.dish.num_layers), 4))
    #     for i in range(self.dish.num_layers):
    #         layer_cmap = ListedColormap([[1, 1, 1], colors[i]])
    #         axs[i].imshow(self.layers_proprioceptive[i], cmap=layer_cmap)
    #         axs[i].set_title(f'Layer - Proprioceptive (Agent) View{i}')
    #         axs[i].axis('off')
                
    #     plt.tight_layout()
    #     plt.show()

    def _initialize_agents(self, num_agents, start_logic, start_positions, agent_archs):
        if start_positions:
            positions = start_positions
        else:
            s = self.dish.size
            mid, q = s // 2, s // 4
            logic_map = {
                'random': [tuple(np.random.randint(0, s, 2)) for _ in range(num_agents)],
                'center': [(mid, mid)] * num_agents,
                'cardinal': [(mid, 0), (mid, s-1), (0, mid), (s-1, mid)],
                'quadrants': [(q, q), (q, s-q), (s-q, q), (s-q, s-q)],
                'corners': [(0, 0), (0, s-1), (s-1, 0), (s-1, s-1)]
            }
            if start_logic not in logic_map:
                raise ValueError(f"Unknown start_logic: '{start_logic}'.")
            positions = logic_map[start_logic]

        for i, pos in enumerate(positions):
            if not (0 <= pos[0] < self.dish.size and 0 <= pos[1] < self.dish.size):
                raise ValueError(f"Start position {pos} is out of bounds.")
            self.agents.append({
                'id': i, 'pos': pos, 'heading': np.random.randint(0, 4),
                'agent': ao.Agent._unit() if agent_archs == "unit clam" else ao.Agent(agent_archs)
            })
        self.history.append([agent['pos'] for agent in self.agents])
        
    def _get_agent_action(self, agent):
        if self.random:
            agent_action = np.random.choice(self.ACTIONS)
        else: # run AO Agent
            agent_input = self.dish.get_stimuli(agent['pos'], radius=self.sensory_radius, shape=self.sensory_shape, mode=self.sensory_mode)
            agent_input_binary = np.asarray([])
            for i in agent_input:
                ib = np.zeros(self.sensory_binary_neurons)
                overage = i - self.sensory_binary_neurons
                if overage < 0:
                    i = i - overage
                ib[0:i] = 1
                agent_input_binary = np.append(agent_input_binary, ib)
    
            agent_action_binary = agent['agent'].next_state(agent_input_binary, INSTINCTS=self.INSTINCTS)
            
            if agent_action_binary == [1]:
                agent_action = self.ACTIONS[0] # move forward
            else: # if [0]
                agent_action = np.random.choice(self.ACTIONS[1:]) # turn right or left randomly
            
        return agent_action

    def set_agent_hyperparameters(self, C_impression_initial = 10, C_impression_match = 1, C_pruning = 1, C_pruning_cutoff = 5):
        for agent in self.agents:
            a = agent['agent']
            a.C_impression_initial = C_impression_initial # strength of impression when first added to C_info
            a.C_impression_match   = C_impression_match  # increment of impression in C_info if accessed
            a.C_pruning            = C_pruning # decrement of impression in C_info if not accessed
            a.C_pruning_cutoff     = C_pruning_cutoff # value below which impressions are pruned from C_info

    def run_step(self):
        """Runs a single step of the experiment for all agents."""
        current_step_positions = []
        for agent in self.agents:
            action = self._get_agent_action(agent)
            x, y = agent['pos']
            if action == 'move_forward':
                dy, dx = self.HEADINGS[agent['heading']]
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.dish.size and 0 <= new_y < self.dish.size:
                    agent['pos'] = (new_x, new_y)
            elif action == 'turn_right':
                agent['heading'] = (agent['heading'] + 1) % 4
            elif action == 'turn_left':
                agent['heading'] = (agent['heading'] - 1 + 4) % 4
            current_step_positions.append(agent['pos'])
        self.history.append(current_step_positions)

    def visualize(self, agents_to_show=None, interval=200, show_paths=True, mode='window'):
        """
        Visualize the assay, animating agent paths as they develop.

        This method cycles through the simulation history step-by-step,
        displaying paths for all agents or a specified subset.

        Parameters:
        - agents_to_show: list of int, optional. A list of agent IDs to
                          display. If None (default), all agents are shown.
        - interval: int, default 200. The delay between frames in milliseconds.
        - show_paths: bool, default True. If True, displays the trailing path
                      of each agent.
        - mode: str, default 'window'. Determines the output format.
                - 'window': Displays the animation in a Matplotlib window (default).
                - 'inline': Returns an HTML object for embedding in notebooks.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        background = self.dish.visualize(combined_only=True)
        ax.imshow(background)
        ax.axis('off')

        if agents_to_show is None:
            agent_ids = list(range(len(self.agents)))
        else:
            num_agents_total = len(self.agents)
            if any(i < 0 or i >= num_agents_total for i in agents_to_show):
                raise ValueError("An invalid agent ID was provided in agents_to_show.")
            agent_ids = agents_to_show

        if not agent_ids:
            plt.close(fig)
            return

        full_agent_colors = plt.cm.jet(np.linspace(0, 1, len(self.agents)))
        history_np = np.array(self.history)
        paths_to_show = history_np[:, agent_ids, :]
        
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
                    path_data = paths_to_show[:frame + 1, idx, :]
                    line.set_data(path_data[:, 0], path_data[:, 1])
            
            current_positions = paths_to_show[frame, :, :]
            agent_scatter.set_offsets(current_positions)
            agent_scatter.set_color([full_agent_colors[i] for i in agent_ids])
            title.set_text(f'Assay State at Step {frame}')
            
            return path_lines + [agent_scatter, title]

        anim = FuncAnimation(fig, update, frames=len(self.history), init_func=init,
                             interval=interval, blit=True)
        
        if mode == 'inline':
            plt.close(fig) # Prevent static plot from showing in notebooks
            display(HTML(anim.to_jshtml()))
        elif mode == 'window':
            plt.show()
        else:
            plt.close(fig)
            raise ValueError("Invalid mode specified. Choose 'window' or 'inline'.")

    # Function to generate dummy data for demonstration
    def export_dummy_data(time_steps=100):
        time = np.arange(time_steps)
        # Response to stimuli: random walk-like data
        response = np.random.randn(time_steps).cumsum()
        # Neuronal response: AO/Projects/petri_dish/dashboard.pysimulated neuron activations
        neuronal = np.sin(time / 10) + np.random.normal(0, 0.5, time_steps)
        # Learning events: sparse events (e.g., 1 when learning occurs)
        learning = np.random.choice([0, 1], size=time_steps, p=[0.9, 0.1])
        # Experience states: states in grid (e.g., position or state value)
        states = np.random.randint(0, 10, time_steps)  # Assume 10 states
        return pd.DataFrame({
            'Time': time,
            'Response to Stimuli': response,
            'Neuronal Response': neuronal,
            'Learning Events': learning,
            'Experience States': states
        })


# # below maintained for testing

# # --- DEMONSTRATION OF PETRI DISH CLASS ---

# # Create a Petri dish instance
# dish = PetriDish(size=50, num_layers=3)
# dish.visualize()

# # Example with custom position
# custom_dist = [
#     {'type': 'radial', 'position': 'random', 'min_p': 0.0, 'max_p': 0.5},
#     {'type': 'radial', 'position': 'top-left', 'min_p': 0.0, 'max_p': 0.5},
#     {'type': 'radial', 'position': (0.25, 0.75), 'min_p': 0.0, 'max_p': 0.5}
# ]
# custom_dish = PetriDish(size=50, num_layers=3, distributions=custom_dist)
# custom_dish.visualize()

# coords_to_check = (25, 25)
# print(f"--- Checking stimuli at coordinate {coords_to_check} ---\n")

# # 1. Original behavior: get value at the exact coordinate
# stimuli_at_point = dish.get_stimuli(coords_to_check)
# print(f"1. Exact stimuli at point: {stimuli_at_point}\n")

# # 2. Uniform concentration within a circular area
# concentration_uniform = dish.get_stimuli(coords_to_check, radius=10, shape='circle', weighting='uniform')
# print(f"2. Uniform concentration (circle, r=10): {concentration_uniform}\n")

# # 3. Uniform concentration within a square area
# concentration_square = dish.get_stimuli(coords_to_check, radius=10, shape='square', weighting='uniform')
# print(f"3. Uniform concentration (square, r=10): {concentration_square}\n")

# # 4. Linear distance-weighted concentration
# concentration_linear = dish.get_stimuli(coords_to_check, radius=10, weighting='linear')
# print(f"4. Linear weighted concentration (r=10): {concentration_linear}\n")

# # 5. Gaussian distance-weighted concentration
# concentration_gaussian = dish.get_stimuli(coords_to_check, radius=10, weighting='gaussian')
# print(f"5. Gaussian weighted concentration (r=10): {concentration_gaussian}\n")

# # 6. Gaussian concentration with a smaller sigma (weights fall off faster)
# concentration_gaussian_tight = dish.get_stimuli(coords_to_check, radius=10, weighting='gaussian', sigma=2)
# print(f"6. Gaussian weighted concentration (r=10, sigma=2): {concentration_gaussian_tight}\n")

# # 7. COUNT mode
# count_circle = dish.get_stimuli(coords_to_check, radius=10, shape='circle', mode='count')
# print(f"7.1. Stimuli COUNT (circle, r=10): {count_circle}\n")
# count_square = dish.get_stimuli(coords_to_check, radius=10, shape='square', mode='count')
# print(f"7.2. Stimuli COUNT (square, r=10): {count_square}\n")



# # --- DEMONSTRATION OF ASSAY CLASS ---

# # 1. Initialize an Assay with 5 agents starting randomly
# assay = Assay(petri_dish=dish, num_agents=5, start_logic='random')

# # 2. Visualize the initial state
# print("Visualizing the initial state of the assay...")
# assay.visualize()

# # 3. Run the simulation for 50 steps
# num_steps = 50
# print(f"\nRunning the assay for {num_steps} steps...")
# for step in range(num_steps):
#     assay.run_step()
# print("...Done.")

# # 4. Visualize the final state and agent paths
# print("\nVisualizing the final state with agent paths...")
# assay.visualize(show_paths=True)