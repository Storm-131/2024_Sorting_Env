# ---------------------------------------------------------*\
# Title: Simple Environment
# Author: TM 06.2024
# ---------------------------------------------------------*/

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from src.input_generator import RandomInputGenerator, SimpleSeasonalInputGenerator, SeasonalInputGenerator
from utils.plotting import plot_env


class SortingEnvironment(gym.Env):
    """Custom Gym environment for a simple sorting system"""

    def __init__(self, max_steps=50, seed=None, noise_lv=0.2, input="r", action_penalty=0.3, threshold=0.8):
        super(SortingEnvironment, self).__init__()

        # Action space: Discrete values for belt speed [0.1, 0.2, ..., 1.0]
        self.action_space = spaces.Discrete(10)

        # Observation space: Total amount of the input material
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.input = input

        # Initialize Input-Generator
        if self.input == "r":
            self.input_generator = RandomInputGenerator(seed=seed)
        elif self.input == "s3":
            self.input_generator = SimpleSeasonalInputGenerator(seed=seed)
        elif self.input == "s9":
            self.input_generator = SeasonalInputGenerator(seed=seed)

        # print (f"Input Generator Check 1: {self.input_generator.generate_input()}")

        # Environment-specific attributes
        self.current_material_input = [30, 50]       # 30% Material A, 50% Material B
        self.current_material_belt = [40, 20]        # 40% Material A, 20% Material B
        self.current_material_sorting = [75, 25]     # 75% Material A, 25% Material B
        self.container_materials = {"A": 0, "A_False": 0, "B": 0, "B_False": 0}

        self.baseline_accuracy = (0.75, 0.75)
        self.start_speed = 0.6
        self.noise_input = noise_lv

        self.accuracy_belt = list(self.baseline_accuracy)
        self.accuracy_sorter = list(self.baseline_accuracy)
        self.belt_speed = self.start_speed
        self.input_occupancy = sum(self.current_material_input) / 100
        self.belt_occupancy = sum(self.current_material_belt) / 100
        self.belt_speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # How much the accuracy is influenced by speed and occupancy
        self.speed_factor = 1
        self.occupancy_factor = 1

        # How much the reward is influenced by accuracy and speed
        self.reward_factor_speed = 0.5
        self.reward_factor_accuracy = 0.5
        self.reward_factor_container_purity = 0
        self.reward_factor_sorting_purity = 0

        # Threshold for the purity of the material in the containers
        self.threshold = threshold
        
        # Noise level for the accuracy of the belt
        self.noise_accuracy = 0.15

       # Define for every speed (left) the maximum occupancy for 100% accuracy (right)
        self.occupancy_limits = {
            0.1: 1.0,
            0.2: 0.9,
            0.3: 0.8,
            0.4: 0.7,
            0.5: 0.6,
            0.6: 0.5,
            0.7: 0.4,
            0.8: 0.3,
            0.9: 0.2,
            1.0: 0.1,
        }

        # Reward data for plotting
        self.reward_data = {
            'Accuracy': [],
            'Speed': [],
            'Occupancy': [],
            'Reward': [],
        }

        # Maximum number of steps in one episode
        self.max_steps = max_steps
        self.action_penalty = action_penalty
        self.previous_speed = None
        self.current_step = 0
        self.set_seed(seed)

    def reset(self, seed=None):

        if seed is not None:
            self.set_seed(seed)
            self.input_generator.set_seed(seed)

        # Reset environment to initial state
        self.current_material_input = [30, 50]
        self.current_material_belt = [40, 20]
        self.current_material_sorting = [75, 25]
        self.container_materials = {"A": 0, "A_False": 0, "B": 0, "B_False": 0}

        self.accuracy_belt = list(self.baseline_accuracy)
        self.accuracy_sorter = list(self.baseline_accuracy)
        self.belt_speed = self.start_speed
        self.input_occupancy = sum(self.current_material_input) / 100
        self.belt_occupancy = sum(self.current_material_belt) / 100

        self.current_step = 0
        self.previous_speed = None

        self.reward_data = {
            'Accuracy': [],
            'Speed': [],
            'Occupancy': [],
            'Reward': [],
        }
        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one step within the environment.
        - Action is chosen based on info about total amount of input-material
        """
        # Sort the material currently inside the sorting machine
        self.sort_material()

        # Update environment ("Move material to next station")
        self.update_environment()

        # Perform action: Set belt speed
        self.set_belt_speed(self.belt_speeds[action])

        # Update accuracy based on speed and occupancy
        self.update_accuracy()

        # Calculate reward, based on accuracy (container, belt) and speed
        reward = self.calculate_reward()

        # Get the next observation
        obs = self._get_obs()

        # Update step counter and check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Return observation, reward, done, info
        return obs, reward, done, False, {}

    def set_seed(self, seed):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        else:
            np.random.seed()
            random.seed()

    def _get_obs(self):
        """Return the current observation of the environment
        - Total amount of material on the input belt, divided by 100
        - Note: Level of Noise increase with the amount of material on the belt!
        """
        total_amount = sum(self.current_material_input) / 100
        noise = np.random.uniform(-self.noise_input, self.noise_input)  # Generate a random noise between -x% and x%

        total_amount += total_amount * noise  # Apply the noise
        total_amount = round(max(0, min(total_amount, 1)), 2)      # Clamp total_amount between 0 and 1

        obs = np.array([total_amount], dtype=np.float32)

        return obs

    def render(self, mode='human', save=False, show=True, log_dir='./img/log', filename='plot', title='', format="svg"):
        """Render the current state of the environment"""
        plot_env(self.current_material_input, self.current_material_belt, self.current_material_sorting,
                 self.container_materials, self.accuracy_belt, self.accuracy_sorter, self.belt_speed, self.reward_data,
                 self.belt_occupancy, save=save, show=show, log_dir=log_dir, filename=filename, title=title,
                 format=format)

    def set_belt_speed(self, new_belt_speed):
        """Set the speed of the belt to a new value"""
        if 0.1 <= new_belt_speed <= 1.0:
            self.previous_speed = self.belt_speed
            self.belt_speed = new_belt_speed
        else:
            print("Invalid belt speed. It must be between 0.1 and 1.0.")

    def sort_material(self):
        """Sort the material based on the current accuracy of the sorting machine, given it's input material
        - Update the Container's contents with the sorted materials
        - Some material might be sorted incorrectly, given the accuracy of the sorter"""
        material_a_to_sort, material_b_to_sort = self.current_material_sorting

        true_a_sorted = int(material_a_to_sort * self.accuracy_sorter[0])
        true_b_sorted = int(material_b_to_sort * self.accuracy_sorter[1])

        false_a_sorted = material_a_to_sort - true_a_sorted
        false_b_sorted = material_b_to_sort - true_b_sorted

        self.container_materials['A'] += true_a_sorted
        self.container_materials['B'] += true_b_sorted
        self.container_materials['A_False'] += false_a_sorted
        self.container_materials['B_False'] += false_b_sorted

        # Calculate total materials and total incorrectly sorted materials
        total_materials = material_a_to_sort + material_b_to_sort
        total_false_sorted = false_a_sorted + false_b_sorted

        # Calculate mean purity
        mean_purity = round(1 - (total_false_sorted / total_materials) if total_materials > 0 else 0, 2)

        return mean_purity

    def update_accuracy(self):
        """Calculate the accuracy for the given batch of material on the belt
        - Based on the speed and occupancy of the belt
        - Noise of 10-15 % is added to the accuracy
        """
        # Get the occupancy limit for the current belt speed
        max_occupancy_for_100_accuracy = self.occupancy_limits[self.belt_speed]

        # If the belt occupancy is within the limit, set accuracy to 1.0
        if self.belt_occupancy <= max_occupancy_for_100_accuracy:
            accuracy = 1.0
        else:
            # Calculate the excess occupancy beyond the limit
            excess_occupancy = self.belt_occupancy - max_occupancy_for_100_accuracy
            # Linearly decrease accuracy from 1.0 to 0.7 based on the excess occupancy
            abatement_rate = 3  # Adjust this factor to control the steepness
            accuracy = 1.0 - (excess_occupancy * abatement_rate)

        # Add noise to the accuracy
        if self.noise_accuracy != 0:
            Noise = np.random.uniform(0.1, self.noise_accuracy)
        else:
            Noise = 0
            
        # Clamp the accuracy between 0 and 1
        accuracy = max(min(accuracy, 1.0), 0.0)
                
        # Set the accuracy of the belt material
        self.accuracy_belt[0] = max(min(accuracy-Noise, 1.0), 0.0)
        self.accuracy_belt[1] = max(min(accuracy-Noise, 1.0), 0.0)

    def calculate_reward(self):
        """Calculate the reward based on the current state of the environment"""

        # 1. Reward for belt speed (Higher Speed = Higher Productivity = Higher Reward)
        normalized_speed = (self.belt_speed - 0.1) / 0.9
        speed_reward = normalized_speed * self.reward_factor_speed
        
        # 2. Reward for accuracy of the belt (Higher Accuracy = Higher Purity = Higher Reward)
        avg_accuracy = sum(self.accuracy_belt) / len(self.accuracy_belt)
        normalized_accuracy = (avg_accuracy - self.threshold) / (1 - self.threshold)  # Focus: accuracy above threshold
        accuracy_reward = normalized_accuracy * self.reward_factor_accuracy

       # y. Reward Sorting-Purity (Higher Purity = Higher Reward)
        sorting_purity_reward = (1 - self.get_belt_purity_estimate()) * self.reward_factor_sorting_purity

        # x. Reward Container-Purity (Higher Purity = Higher Reward)
        container_purity = self.get_container_purity()
        purity_normalized = (container_purity - self.threshold) / (1 - self.threshold)  # Scaling purity to 0-1
        if purity_normalized < 0:
            container_purity_reward = 0
        else:
            container_purity_reward = purity_normalized * self.reward_factor_container_purity

        # Add penalty for changing the belt speed
        if self.previous_speed is not None and self.belt_speed != self.previous_speed:
            penalty = self.action_penalty
        else:
            penalty = 0

     # Add penalty for low accuracy or low purity ("threshold")
        if avg_accuracy < self.threshold:
            total_reward = -0.1
        else:
            # Calculate total reward
            total_reward = container_purity_reward + sorting_purity_reward + accuracy_reward + speed_reward - penalty

        # Save reward data for plotting
        self.reward_data['Reward'].append(round(total_reward, 2))
        self.reward_data['Accuracy'].append(avg_accuracy)
        self.reward_data['Speed'].append(self.belt_speed)
        self.reward_data['Occupancy'].append(self.belt_occupancy)

        return total_reward

    def update_environment(self):
        """Update the environment by moving the material to the next station"""
        self.current_material_sorting = self.current_material_belt
        self.current_material_belt = self.current_material_input
        self.belt_occupancy = self.input_occupancy
        a, b, self.input_occupancy, _ = self.input_generator.generate_input()
        self.current_material_input = [a, b]
        self.accuracy_sorter = self.accuracy_belt.copy()

    def get_belt_purity_estimate(self):
        """Calculate the overall error rate based on the current accuracies and material amounts."""
        material_a = self.current_material_belt[0]
        material_b = self.current_material_belt[1]

        true_a = material_a * self.accuracy_belt[0]
        true_b = material_b * self.accuracy_belt[1]

        false_a = material_a - true_a
        false_b = material_b - true_b

        total_material = material_a + material_b
        total_false = false_a + false_b

        if total_material == 0:
            error_rate = 0
        else:
            error_rate = total_false / total_material

        return error_rate

    def get_container_purity(self):
        """Calculate the ratio of correct material in the containers as reference for the reward"""
        # Calculate total material in the containers
        total_material = sum(self.container_materials.values())

        # Calculate the total correct material (A and B)
        correct_material = self.container_materials['A'] + self.container_materials['B']

        # Calculate the percentage of correct material
        if total_material == 0:
            correct_ratio = 0  # To avoid division by zero
        else:
            correct_ratio = (correct_material / total_material)

        return round(correct_ratio, 2)

    def print_possible_actions(self):
        """Print all possible actions that can be taken in the environment"""
        # Generate all possible actions
        actions = self.belt_speeds

        # Documenting all possible actions
        action_meanings = {}
        for i, action in enumerate(actions):
            action_meanings[i] = {"belt_speed": action}

        print("Documenting all possible actions:")
        for action, details in action_meanings.items():
            print(f"Action {action}: Belt Speed = {details['belt_speed']}")

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
