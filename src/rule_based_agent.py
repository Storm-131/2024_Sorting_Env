# ---------------------------------------------------------*\
# Title: Rule-Based Agent #1 (Simple Environment)
# Author: TM 2024
# ---------------------------------------------------------*/

import itertools
import pandas as pd
import numpy as np
import csv
import os

# ---------------------------------------------------------*/
# Rule-Based Agent #1 (Simple Environment)
# ---------------------------------------------------------*/


class Rule_Based_Agent_simple:

    def __init__(self, env):

        self.env = env
        # Important: RBA can't be trained on time-dependent data, hence it calculates it's reward
        # based on the current state of "Belt Accuracy" (not the accumulative Container-Purity)
        # self.env.reward_factor_purity = 0           # Reward Factor for Purity (Time-Dependent Data)
        # self.env.reward_factor_accuracy = 0.5       # Reward Factor for Accuracy

        self.reward_table = self.create_reward_table()

    def create_reward_table(self):
        """Create a table that calculates the reward for each possible belt-occupancy and speed."""
        reward_table = []
        for speed in self.env.belt_speeds:
            for occupancy in range(0, 101):  # Assume 100 different occupancy levels
                reward = self.calculate_reward(occupancy, speed)
                reward_table.append((occupancy, speed, reward))
        return reward_table

    def calculate_reward(self, occupancy, speed):
        """Calculate the reward based on given occupancy and speed using the environment."""
        # Set the environment's state based on occupancy
        self.env.belt_occupancy = occupancy / 100
        self.env.set_belt_speed(speed)
        self.env.update_accuracy()

        # Calculate the reward using the environment's method
        reward = self.env.calculate_reward()
        return reward

    def get_best_action(self, current_occupancy):
        """Get the action with the highest reward for the current occupancy."""
        best_action = None
        best_reward = -float('inf')
        for speed in self.env.belt_speeds:
            reward = self.get_reward_from_table(current_occupancy, speed)
            if reward > best_reward:
                best_reward = reward
                best_action = speed
        # print(f"Occupancy: {current_occupancy}, Best Speed: {best_action}, Predicted Reward: {best_reward}")
        return best_action

    def get_reward_from_table(self, occupancy, speed):
        """Get the reward from the reward table for a given occupancy and speed."""
        for row in self.reward_table:
            if row[0] == occupancy and row[1] == speed:
                return row[2]
        return -float('inf')  # Default reward if no match is found

    def predict(self, obs, deterministic=True):
        """Given an observation, choose the best action."""
        current_occupancy = int(round(obs[0] * 100))  # Use round instead of int
        best_speed = self.get_best_action(current_occupancy)
        action = self.env.belt_speeds.index(best_speed)
        if action is None:
            action = self.env.action_space.sample()
        return action, {}

    def save_reward_table(self, folder='./log', filename='reward_table.csv'):
        """Save the reward table to a CSV file."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Occupancy', 'Speed', 'Reward'])
            for row in self.reward_table:
                writer.writerow(row)
        print(f"Reward table saved to {filepath}")

    def create_and_save_accuracy_matrix(self, occupancy_values, speed_values, folder='./log', filename='accuracy_matrix.csv'):
        """Create an accuracy matrix and save it to a CSV file."""
        accuracy_matrix = np.zeros((len(occupancy_values), len(speed_values)))
        for i, occupancy in enumerate(occupancy_values):
            for j, speed in enumerate(speed_values):
                self.env.belt_occupancy = occupancy / 100
                self.env.set_belt_speed(speed)
                self.env.update_accuracy()
                accuracy_matrix[i, j] = sum(self.env.accuracy_belt) / len(self.env.accuracy_belt)

        df = pd.DataFrame(accuracy_matrix, index=occupancy_values, columns=speed_values)
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        df.to_csv(filepath)
        print(f"Accuracy matrix saved to {filepath}")
        return accuracy_matrix

    def extract_data_from_reward_table(self):
        """Extract data from the reward table for plotting."""
        occupancies = np.array([row[0] for row in self.reward_table])
        speeds = np.array([row[1] for row in self.reward_table])
        rewards = np.array([row[2] for row in self.reward_table])
        occupancy_values = np.unique(occupancies)
        speed_values = np.unique(speeds)
        reward_matrix = rewards.reshape((len(speed_values), len(occupancy_values))).T
        return occupancy_values, speed_values, reward_matrix

    def save_sorted_reward_table(self, reward_matrix, occupancy_values, folder='./log', filename='reward_table_sorted.csv'):
        """Save the sorted reward table with headers."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Occupancy', 'Speed_10', 'Speed_20', 'Speed_30', 'Speed_40',
                            'Speed_50', 'Speed_60', 'Speed_70', 'Speed_80', 'Speed_90', 'Speed_100'])
            for index, row in enumerate(reward_matrix):
                writer.writerow([occupancy_values[index]] + list(row))
        print(f"Sorted reward table saved to {filepath}")

    def run_analysis(self):
        """Run the full analysis pipeline."""
        self.save_reward_table()
        occupancy_values, speed_values, reward_matrix = self.extract_data_from_reward_table()
        self.save_sorted_reward_table(reward_matrix, occupancy_values)
        self.create_and_save_accuracy_matrix(occupancy_values, speed_values)
        # self.plot_results(occupancy_values, speed_values, accuracy_matrix, reward_matrix)

# ---------------------------------------------------------*/
# Rule-Based Agent #2 (Advanced Environment)
# ---------------------------------------------------------*/


class Rule_Based_Agent_adv:

    def __init__(self, env):
        self.env = env
        # Important: RBA can't be trained on time-dependent data, hence it calculates its reward
        # based on the current state of "Belt Accuracy" (not the accumulative Container-Purity)
        # self.reward_factor_sorting_purity = 0.5      # Reward Factor for Purity (Time-Dependent Data)
        # self.env.reward_factor_accuracy = 0.5        # Reward Factor for Accuracy

        self.reward_table = self.create_reward_table()

    def create_reward_table(self):
        """Create a table that calculates the reward for each possible belt-occupancy, speed, mode, and ratio category."""
        reward_table = []
        for speed in self.env.belt_speeds:
            for mode in self.env.sorting_modes:
                for occupancy in range(0, 101):  # Assume 100 different occupancy levels
                    for ratio_category in [0, 1, 2]:  # Assume ratio categories 0, 1, 2
                        reward = self.calculate_reward(occupancy, speed, mode, ratio_category)
                        reward_table.append((occupancy, speed, mode, ratio_category, reward))
        return reward_table

    def calculate_reward(self, occupancy, speed, mode, ratio_category):
        """Calculate the reward based on given occupancy, speed, mode, and ratio category using the environment."""
        # Set the environment's state based on occupancy and ratio category
        self.env.belt_occupancy = occupancy / 100
        self.env.set_belt_speed(speed)

        # Simulate the ratio category in the environment
        if ratio_category == 0:
            self.env.current_material_belt = [50, 50]  # Example for balanced ratio
        elif ratio_category == 1:
            self.env.current_material_belt = [75, 25]  # Example for more A
        elif ratio_category == 2:
            self.env.current_material_belt = [25, 75]  # Example for more B

        self.env.update_accuracy(mode=mode)

        # Calculate the reward using the environment's method
        reward = self.env.calculate_reward(mode=mode)
        return reward

    def get_best_action(self, current_occupancy, current_ratio):
        """Get the best action for the current occupancy and ratio category."""
        best_action = None
        best_reward = -float('inf')
        mode = self.get_mode_from_ratio(current_ratio)
        # print(f"Selected mode for ratio {current_ratio}: {mode}")  # Debugging information
        for speed in self.env.belt_speeds:
            reward = self.get_reward_from_table(current_occupancy, speed, mode, current_ratio)
            if reward > best_reward:
                best_reward = reward
                best_action = (speed, mode)
        return best_action

    def get_reward_from_table(self, occupancy, speed, mode, ratio_category):
        """Get the reward from the reward table for a given occupancy, speed, mode, and ratio category."""
        for row in self.reward_table:
            if row[0] == occupancy and row[1] == speed and row[2] == mode and row[3] == ratio_category:
                return row[4]
        return -float('inf')  # Default reward if no match is found

    def get_mode_from_ratio(self, ratio):
        """Get the sorting mode based on the ratio category."""
        # print(f"Received ratio in get_mode_from_ratio: {ratio}")  # Debugging information
        if ratio == 0:
            return 0
        elif ratio == 1:
            return 1
        elif ratio == 2:
            return 2
        else:
            print(f"Invalid ratio value: {ratio}")
            raise ValueError("Invalid ratio value")

    def predict(self, obs, deterministic=True):
        """Given an observation, choose the best action."""
        current_occupancy = int(round(obs[0] * 100))  # Use round instead of int
        current_ratio = obs[1]
        # print(f"Observation - Occupancy: {current_occupancy}, Ratio: {current_ratio}")  # Debugging information
        best_speed, best_mode = self.get_best_action(current_occupancy, current_ratio)
        action = self.env.belt_speeds.index(best_speed) * len(self.env.sorting_modes) + best_mode
        if action is None:
            action = self.env.action_space.sample()
        # print(f"Predicted action - Speed: {best_speed}, Mode: {best_mode}, Action: {action}")  # Debugging information
        return action, {}

    def save_reward_table(self, folder='./log', filename='reward_table_adv.csv'):
        """Save the reward table to a CSV file."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Occupancy', 'Speed', 'Mode', 'Ratio', 'Reward'])
            for row in self.reward_table:
                writer.writerow(row)
        print(f"Reward table saved to {filepath}")

    def create_and_save_accuracy_matrix(self, occupancy_values, speed_values, folder='./log', filename='accuracy_matrix_adv.csv'):
        """Create an accuracy matrix and save it to a CSV file."""
        accuracy_matrix = np.zeros((len(occupancy_values), len(speed_values), len(self.env.sorting_modes)))
        for i, occupancy in enumerate(occupancy_values):
            for j, speed in enumerate(speed_values):
                for k, mode in enumerate(self.env.sorting_modes):
                    self.env.belt_occupancy = occupancy / 100
                    self.env.set_belt_speed(speed)
                    self.env.update_accuracy(mode=mode)
                    accuracy_matrix[i, j, k] = sum(self.env.accuracy_belt) / len(self.env.accuracy_belt)

        df = pd.DataFrame(accuracy_matrix.reshape(len(occupancy_values), -1), index=occupancy_values,
                          columns=pd.MultiIndex.from_product([speed_values, self.env.sorting_modes], names=['Speed', 'Mode']))
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        df.to_csv(filepath)
        print(f"Accuracy matrix saved to {filepath}")
        return accuracy_matrix

    def extract_data_from_reward_table(self):
        """Extract data from the reward table for plotting."""
        occupancies = np.array([row[0] for row in self.reward_table])
        speeds = np.array([row[1] for row in self.reward_table])
        modes = np.array([row[2] for row in self.reward_table])
        ratios = np.array([row[3] for row in self.reward_table])
        rewards = np.array([row[4] for row in self.reward_table])
        occupancy_values = np.unique(occupancies)
        speed_values = np.unique(speeds)
        mode_values = np.unique(modes)
        ratio_values = np.unique(ratios)
        reward_matrix = rewards.reshape((len(ratio_values), len(mode_values), len(
            speed_values), len(occupancy_values))).transpose(3, 2, 1, 0)
        return occupancy_values, speed_values, mode_values, ratio_values, reward_matrix

    def save_sorted_reward_table(self, reward_matrix, occupancy_values, folder='./log', filename='reward_table_sorted_adv.csv'):
        """Save the sorted reward table with headers."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Occupancy'] + [f'Speed_{speed}_Mode_{mode}_Ratio_{ratio}' for speed, mode,
                                       ratio in itertools.product(self.env.belt_speeds, self.env.sorting_modes, [0, 1, 2])]
            writer.writerow(headers)
            for index, row in enumerate(reward_matrix.reshape(len(occupancy_values), -1)):
                writer.writerow([occupancy_values[index]] + list(row))
        print(f"Sorted reward table saved to {filepath}")

    def run_analysis(self):
        """Run the full analysis pipeline."""
        self.save_reward_table()
        occupancy_values, speed_values, mode_values, ratio_values, reward_matrix = self.extract_data_from_reward_table()
        self.save_sorted_reward_table(reward_matrix, occupancy_values)
        self.create_and_save_accuracy_matrix(occupancy_values, speed_values)
        # self.plot_results(occupancy_values, speed_values, accuracy_matrix, reward_matrix)


# -------------------------Notes-----------------------------------------------*\
# This Rule-Based Agent class is designed to work with the SortingEnv environment.
# -----------------------------------------------------------------------------*\
