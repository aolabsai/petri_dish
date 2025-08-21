
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
    def __init__(self, petri_dish, num_agents=5, start_logic='random', start_positions=None, agent_archs="unit clam", assay_loadagents=""):
        steps = 1000
        metainfo = 7
        self.dish = petri_dish
        
        if type(assay_loadagents) is Assay:
            # load number of agents from inputted assay
            print("-----AGENTS LOADED FROM SUPPLIED ASSAY-----")
            self.num_agents = assay_loadagents.num_agents
            self.agent_archs = assay_loadagents.agent_archs
        else:
            self.num_agents = num_agents
            self.agent_archs = agent_archs
        self.num_learning_types = len(self.agent_archs.arch_c)
        self.history = np.zeros((steps, self.num_agents, 2), dtype=int) # Shape: (steps, agents, (x,y))
        self.meta_history = np.zeros((steps, self.num_agents, metainfo+self.num_learning_types), dtype=object) # Shape: (steps, agents, meta_features)
        
        self._initialize_agents(start_logic, start_positions, assay_loadagents)
        self.step = 0 # Initialize step counter

        self.random = False # set to true to move agents randomly, not through ao.next_state, useful for debugging the UI in isolation 
        self.INSTINCTS = True
        self.sensory_binary_neurons = 10
        self.sensory_radius = 3
        self.sensory_shape = "circle"
        self.sensory_mode = "count"

        self.ACTIONS = ['move_forward', 'turn_right', 'turn_left']
        self.HEADINGS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)} # N, E, S, W in (y,x)


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
                # load agent object from inputted assay
                agent = assay_loadagents.agents[i]['agent'] # loading agent from previous assay
            else:
                agent = ao.Agent(self.agent_archs)
    
                # setting up counters for learning events
                for c in self.agent_archs.C_types_names:
                    setattr(agent, c, 0)

            self.agents[i] = {
                'id': i,
                'pos': pos,
                'heading': np.random.randint(0, 4),
                'agent': agent
            }
            # Store initial position in history
            self.history[0, i, 0] = pos[0] # x
            self.history[0, i, 1] = pos[1] # y
        

    def _get_agent_action(self, agent):
        if self.random:
            return np.random.choice(self.ACTIONS)
        
        # run AO Agent
        stimuli_input = self.dish.get_stimuli(agent['pos'], shape=self.sensory_shape, radius=self.sensory_radius,  mode=self.sensory_mode)
        agent_input_binary = []
        for val in stimuli_input:
            input_binary = np.zeros(self.sensory_binary_neurons)
            input_binary[0:val] = 1
            agent_input_binary.extend(input_binary)
        agent_input_binary = np.array(agent_input_binary)

        agent_action_binary = agent['agent'].next_state(agent_input_binary, INSTINCTS=self.INSTINCTS, print_result=True)
        
        if np.array_equal(agent_action_binary, [1]):
            return self.ACTIONS[0]  # move forward
        else:
            return np.random.choice(self.ACTIONS[1:]) # turn right or left
            
    def set_agent_hyperparameters(self, C_impression_initial=10, C_impression_match=1, C_pruning=1, C_pruning_cutoff=5):
        for agent_dict in self.agents:
            a = agent_dict['agent']
            a._save_C_info = True
            a.C_impression_initial = C_impression_initial
            a.C_impression_match = C_impression_match
            a.C_pruning = C_pruning
            a.C_pruning_cutoff = C_pruning_cutoff


    def pretrain_random(self, steps):
        """Pre-train agents on random data using built-in ao_core methods; this helps give the agent a greater range of motion before lessons from instincts kick in"""
        for i, agent_dict in enumerate(self.agents):
            for s in range(steps):
                agent_dict['agent'].reset_state(training=True)
                agent_dict['agent'].reset_state()


    def run_step(self, steps):
        """Runs a single step of the experiment for all agents."""

        if self.step == self.history.shape[0]:
            print("Warning: History limit reached, assay memory full.")
            return

        for s in range(steps):
            self.step += 1

            for i, agent_dict in enumerate(self.agents):
                action = self._get_agent_action(agent_dict)
                x, y = agent_dict['pos']
                if action == 'move_forward':
                    dy, dx = self.HEADINGS[agent_dict['heading']]
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.dish.size and 0 <= new_y < self.dish.size:
                        agent_dict['pos'] = (new_x, new_y)
                elif action == 'turn_right':
                    agent_dict['heading'] = (agent_dict['heading'] + 1) % 4
                elif action == 'turn_left':
                    agent_dict['heading'] = (agent_dict['heading'] - 1 + 4) % 4

                # Store history for the current step
                self.history[self.step, i, 0] = agent_dict['pos'][0] # x
                self.history[self.step, i, 1] = agent_dict['pos'][1] # y
                
                # For AO agents, store meta history
                agent_instance = agent_dict['agent']
                self.meta_history[self.step, i, 0] = sum(agent_instance.astate[0, agent_instance.arch.I[0]])
                self.meta_history[self.step, i, 1] = sum(agent_instance.astate[0, agent_instance.arch.I[1]])
                self.meta_history[self.step, i, 2] = sum(agent_instance.astate[0, agent_instance.arch.I[2]])
                self.meta_history[self.step, i, 3] = sum(agent_instance.astate[0, agent_instance.arch.Q[0]])
                self.meta_history[self.step, i, 4] = sum(agent_instance.astate[0, agent_instance.arch.Q[1]])
                self.meta_history[self.step, i, 5] = sum(agent_instance.astate[0, agent_instance.arch.Q[2]])
                try: self.meta_history[self.step, i, 6] = agent_instance.neurons[agent_instance.arch.Z__flat[0]].outputs.size
                except: AttributeError 
                for c in range(self.num_learning_types):
                    self.meta_history[self.step, i, 7+c] = getattr(agent_instance, self.agent_archs.C_types_names[c])


    def visualize(self, agents_to_show=None, interval=200, show_paths=True, mode='window'):
        # This function seems mostly fine, but the history slicing needs to be updated.
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

    def export_data(self, to_step=None, file_name="assay_data.pkl", env_file_name="gridworld.pkl"):
        import pickle
        import pandas as pd
        
        if to_step is None: 
            to_step = self.step + 1 # Include the latest step
        
        agent_names = []
        agent_dfs = []
        for a in range(self.num_agents):
            agent_names.append(f"Agent_{a}") # Add an agent identifier
            df = pd.DataFrame({
                'Time': np.arange(to_step),
                'stimuli0': self.meta_history[:to_step, a, 0],
                'stimuli1': self.meta_history[:to_step, a, 1],
                'stimuli2': self.meta_history[:to_step, a, 2],
                'neuronal0':   self.meta_history[:to_step, a, 3],
                'neuronal1':   self.meta_history[:to_step, a, 4],
                'neuronal2':   self.meta_history[:to_step, a, 5],
                'Experience States':   self.meta_history[:to_step, a, 6],
                'pos_x': self.history[:to_step, a, 0],
                'pos_y': self.history[:to_step, a, 1]
            })
            for c in range(self.num_learning_types):
                df[self.agent_archs.C_types_names[c]] = self.meta_history[:to_step, a, 7+c]
            agent_dfs.append(df)
        
        # Combine all agent DataFrames into a single one, and save the env layers, all in a list for easy exporting

        agent_data = [ agent_names, agent_dfs ]
        env_data = self.dish.layers

        final_data = [ agent_data, env_data ]

        try:
            with open(file_name, "wb") as f:
                pickle.dump(final_data, f)
            print(f"DataFrame successfully saved to {file_name}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

        return final_data