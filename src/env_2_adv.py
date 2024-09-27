# ---------------------------------------------------------*\
# Title: Advanced Environment
# Author: TM 06.2024
# ---------------------------------------------------------*/

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from src.input_generator import RandomInputGenerator, SimpleSeasonalInputGenerator, SeasonalInputGenerator
from utils.plotting import plot_env
import itertools


class SortingEnvironmentAdv(gym.Env):
    """Custom Gym environment for a sorting system with advanced handling of input ratios"""

    def __init__(self, max_steps=50, seed=None, noise_lv=0.2, input="r", action_penalty=0.3, threshold=0.8):
        super(SortingEnvironmentAdv, self).__init__()

        # Action space: Discrete values for belt speed [0.1,..,1.0 ]and sorting mode [0, 1, 2]
        self.action_space = spaces.Discrete(30)  # 10 speeds x 3 sorting modes = 30 actions

        # Observation space: Total amount of the input material and its ratio (A/B)
        self.observation_space = spaces.Box(low=0, high=2, shape=(2,), dtype=np.float32)

        self.input = input

        # Initialize Input-Generator
        if self.input == "r":
            self.input_generator = RandomInputGenerator(seed=seed)
        elif self.input == "s3":
            self.input_generator = SimpleSeasonalInputGenerator(seed=seed)
        elif self.input == "s9":
            self.input_generator = SeasonalInputGenerator(seed=seed)

        # print(f"Input Generator Check 1: {self.input_generator.generate_input()}")

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
        self.sorting_modes = [0, 1, 2]  # 0: Basic Sorting, 1: Positive (Sorting for A), 2: Negative (Sorting for B)

        # How much the accuracy is influenced by speed and occupancy
        self.speed_factor = 1
        self.occupancy_factor = 1

        # How much the reward is influenced by accuracy and speed
        self.reward_factor_speed = 0.5
        self.reward_factor_accuracy = 0.5
        self.reward_factor_container_purity = 0
        self.reward_factor_sorting_purity = 0

        # Threshold for the purity of the containers
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
            'Modes': [],  # Track the sequence of modes used
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
            'Modes': [],
        }
        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one step within the environment.
        - Action is chosen based on info about total amount and ratio of input-material
        """
        # Sort the material currently inside the sorting machine
        self.sort_material()

        # Update environment ("Move material to next station")
        self.update_environment()

        # Convert discrete action to corresponding belt speed and sorting mode
        speed_action = action // 3          # 10 different belt speeds, changes every 3 actions
        sorting_action = action % 3         # 3 different sorting modes, changes every 1 action

        # print(f"Speed Action: {speed_action}, Sorting Action: {sorting_action}")

        # Perform action: Set belt speed
        self.set_belt_speed(self.belt_speeds[speed_action])

        # Update accuracy based on speed, occupancy, and sorting mode
        self.update_accuracy(mode=sorting_action)

        # Calculate reward, based on accuracy (container, belt) and speed
        reward = self.calculate_reward(mode=sorting_action)

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
        - A value encoding the category of the input material ratio (A/B)
            - See get_ratio() for details
        - Note: Level of Noise increase with the amount of material on the belt!
        """
        total_amount = sum(self.current_material_input) / 100
        noise = np.random.uniform(-self.noise_input, self.noise_input)  # Generate a random noise between -x% and x%

        total_amount += total_amount * noise  # Apply the noise
        total_amount = round(max(0, min(total_amount, 1)), 2)      # Clamp total_amount between 0 and 1

        _, ratio_category = self.get_ratio(material=self.current_material_input)

        obs = np.array([total_amount, ratio_category], dtype=np.float32)

        return obs

    def render(self, mode='human', save=False, show=True, log_dir='./img/log', filename='plot', title='', format="svg"):
        """Render the current state of the environment"""
        plot_env(self.current_material_input, self.current_material_belt, self.current_material_sorting,
                 self.container_materials, self.accuracy_belt, self.accuracy_sorter, self.belt_speed, self.reward_data,
                 self.belt_occupancy, save=save, show=show, log_dir=log_dir, filename=filename, title=title,
                 format=format, sorting_mode=True)

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
        mean_purity = 1 - (total_false_sorted / total_materials) if total_materials > 0 else 0

        return mean_purity

    def update_accuracy(self, mode=0):
        """Calculate the accuracy for the given batch of material on the belt
        - Based on the speed and occupancy of the belt
        - Adjust the accuracy + noise based on the sorting mode
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

        if self.noise_accuracy != 0:
            Noise = np.random.uniform(0.1, self.noise_accuracy)
        else:
            Noise = 0

        # ---------------------------------------------------------*/
        # New: Sorting Mode Section
        # ---------------------------------------------------------*/
        # Check the ratio of the material on the belt and determine the sorting category
        _, ratio_category = self.get_ratio(noise=False, noise_level=0.05, material=self.current_material_belt)

        #  1) Basic Sorting:
        if mode == 0:
            if ratio_category == 0:
                accuracy += 0.15    
                Noise = np.random.uniform(0, 0.05)
            else:
                accuracy -= 0.1

        # 2) Positive Sorting (More A -> Higher Accuracy for A) ratio A/B >= 3:
        elif mode == 1:
            if ratio_category == 1:
                accuracy += 0.15                    # Increase in Overall-Accuracy
                Noise = np.random.uniform(0, 0.05)  # Noise Reduction
            else:
                accuracy -= 0.1

        # 3) Negative Sorting (More B -> Higher Accuracy for B) ratio A/B <= 1/3
        elif mode == 2:
            if ratio_category == 2:
                accuracy += 0.15                    # Increase in Overall-Accuracy
                Noise = np.random.uniform(0, 0.05)  # Noise Reduction
            else:
                accuracy -= 0.1
        # ---------------------------------------------------------*/
        
        # Clamp the accuracy between 0 and 1
        accuracy = max(min(accuracy, 1.0), 0.0)

        # Set the accuracy of the belt material
        self.accuracy_belt[0] = max(min(accuracy-Noise, 1.0), 0.0)
        self.accuracy_belt[1] = max(min(accuracy-Noise, 1.0), 0.0)

    def calculate_reward(self, mode=None):
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
        self.reward_data['Modes'].append(mode)      # New: Track the sequence of modes used

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

    def get_ratio(self, noise=False, noise_level=0.1, material=None):
        """ Calculate the ratio of two materials (A/B) and determine the sorting category based on the ratio.
        - If noise is enabled, the ratio will be adjusted by a random value between -noise_level and +noise_level
        - The ratio is categorized into three categories, based on the raw ratio:
            - Basic: Similiar amounts of A and B, no change in accuracy
            - Positive: More A than B, higher accuracy for A
            - Negative: More B than A, higher accuracy for B
        """
        if material is None:
            raise ValueError("Material must be provided to calculate the ratio.")
        else:
            A, B = material

        # Calculate the raw ratio
        if B == 0:
            raw_ratio = 3
        else:
            raw_ratio = round(A / B, 2)

        # Apply noise to the ratio if noise is enabled
        if noise:
            raw_ratio = max(0, raw_ratio + random.uniform(-noise_level, noise_level))

        # Determine the category based on the raw ratio
        if raw_ratio >= 3:
            ratio_category = 1    # Positive (More A)
        elif raw_ratio <= 1/3:
            ratio_category = 2     # Negative (More B)
        elif raw_ratio > 1/3 and raw_ratio < 3:
            ratio_category = 0    # Basic (Equal A and B)

        return raw_ratio, ratio_category

    def print_possible_actions(self):
        """Print all possible actions that can be taken in the environment"""
        # Generate all possible actions
        actions = list(itertools.product(self.belt_speeds, self.sorting_modes))

        # Documenting all possible actions
        action_meanings = {}
        for i, action in enumerate(actions):
            action_meanings[i] = {"belt_speed": action[0], "sorting_mode": action[1]}

        print("Documenting all possible actions:")
        for action, details in action_meanings.items():
            print(f"Action {action}: Belt Speed = {details['belt_speed']}, Sorting Mode = {details['sorting_mode']}")


# -------------------------Notes-----------------------------------------------*\
"""
The agent will learn through interaction with the environment which sorting mode to 
choose, based on the observed ratio-category of input-materials:
- When the ratio of A to B is less than or equal to 1/3, it should use positive sorting.
- When the ratio of A to B is bigger than or equal to 3, it should use negative sorting.
- For other cases, it should use basic sorting.

| Action |  (`belt_speed`)                    |  (`sorting_mode`)             |
|--------|------------------------------------|-------------------------------|
| 0      | 0.1                                | 0 (Basic)                     |
| 1      | 0.1                                | 1 (Positive)                  |
| 2      | 0.1                                | 2 (Negative)                  |
| 3      | 0.2                                | 0 (Basic)                     |
| 4      | 0.2                                | 1 (Positive)                  |
| 5      | 0.2                                | 2 (Negative)                  |
| 6      | 0.3                                | 0 (Basic)                     |
| 7      | 0.3                                | 1 (Positive)                  |
| 8      | 0.3                                | 2 (Negative)                  |
| 9      | 0.4                                | 0 (Basic)                     |
| 10     | 0.4                                | 1 (Positive)                  |
| 11     | 0.4                                | 2 (Negative)                  |
| 12     | 0.5                                | 0 (Basic)                     |
| 13     | 0.5                                | 1 (Positive)                  |
| 14     | 0.5                                | 2 (Negative)                  |
| 15     | 0.6                                | 0 (Basic)                     |
| 16     | 0.6                                | 1 (Positive)                  |
| 17     | 0.6                                | 2 (Negative)                  |
| 18     | 0.7                                | 0 (Basic)                     |
| 19     | 0.7                                | 1 (Positive)                  |
| 20     | 0.7                                | 2 (Negative)                  |
| 21     | 0.8                                | 0 (Basic)                     |
| 22     | 0.8                                | 1 (Positive)                  |
| 23     | 0.8                                | 2 (Negative)                  |
| 24     | 0.9                                | 0 (Basic)                     |
| 25     | 0.9                                | 1 (Positive)                  |
| 26     | 0.9                                | 2 (Negative)                  |
| 27     | 1.0                                | 0 (Basic)                     |
| 28     | 1.0                                | 1 (Positive)                  |
| 29     | 1.0                                | 2 (Negative)                  |
"""
# -----------------------------------------------------------------------------*\
