# ---------------------------------------------------------*\
# Title: Plot Environment Statistics
# Author: TM 2024
# ---------------------------------------------------------*/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.env_1_simple import SortingEnvironment
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

# ---------------------------------------------------------*/
# Configuration
# ---------------------------------------------------------*/
occupancies = range(0, 101)
occupancy_categories = np.arange(0, 101, 20)
factors = np.round(np.arange(0.2, 1.0, 0.2), 1)

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16
})

# ---------------------------------------------------------*/
# Accuracy
# ---------------------------------------------------------*/


def calculate_mean_accuracies(env, belt_speeds, occupancies):
    occupancy_data = {}
    occupancies = np.array(occupancies) / 100

    for occupancy in occupancies:
        mean_accuracies = []
        for speed in belt_speeds:
            env.belt_speed = speed
            env.belt_occupancy = occupancy
            env.update_accuracy()
            accuracy = env.accuracy_belt
            mean_accuracy = np.mean(accuracy)
            mean_accuracies.append(mean_accuracy)
        occupancy_data[occupancy] = mean_accuracies

    return occupancy_data


def plot_accuracies(ax, title, belt_speeds, occupancy_data, xlabel='Belt Speed', ylabel='Mean Accuracy'):
    for occupancy, accuracies in occupancy_data.items():
        ax.plot(belt_speeds, accuracies, label=f'Occupancy {occupancy*100:.0f}%')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def generate_single_accuracy_plot(ax, env, belt_speeds, occupancies):

    occupancies = np.array(occupancies) / 100

    for speed in belt_speeds:
        accuracies = []
        for occupancy in occupancies:
            env.belt_speed = speed
            env.belt_occupancy = occupancy
            env.update_accuracy()
            accuracy = env.accuracy_belt
            mean_accuracy = np.mean(accuracy)
            accuracies.append(mean_accuracy)

        ax.plot(occupancies * 100, accuracies, label=f'Speed {speed:.1f}')

    # ax.set_title(
    #    f'Mean Accuracy vs. Belt Speed; Speed Factor={env.speed_factor:.1f}, Occupancy Factor={env.occupancy_factor:.1f}')
    ax.set_xlabel('Occupancy (%)')
    ax.set_ylabel('Mean Accuracy')
    ax.legend(loc='lower left')
    ax.grid(True)
    ax.set_ylim(0.7, 1.02)  # Set y range to 0.7


def generate_accuracy_heatplot(fig, ax, env, belt_speeds):
    occupancies = range(0, 101)
    occupancy_data = calculate_mean_accuracies(env, belt_speeds, occupancies)
    occupancy_values = np.array(list(occupancy_data.keys()))
    speed_values = belt_speeds
    accuracy_matrix = np.zeros((len(occupancy_values), len(speed_values)))

    for i, occupancy in enumerate(occupancy_values):
        accuracy_matrix[i, :] = occupancy_data[occupancy]

    df = pd.DataFrame(accuracy_matrix, index=occupancy_values, columns=speed_values)
    
    # Check if the directory exists, if not, create it
    if not os.path.exists('./log/'):
        os.makedirs('./log/')
        
    df.to_csv('./log/accuracy_matrix_2.csv')

    # Create a custom colormap that includes black for values below Threshold
    viridis = plt.cm.viridis(np.linspace(0, 1, 32))
    viridis[:int(env.threshold*32), :] = [0, 0, 0, 1]  # Set the irrelevant accuracies to black
    custom_cmap = ListedColormap(viridis)

    # Plot the heatmap
    c = ax.imshow(accuracy_matrix, aspect='auto', extent=[0, len(
        speed_values)-1, 0, 1], origin='lower', cmap=custom_cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Accuracy')

    # ax.set_title('Accuracy based on Speed and Occupancy')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Occupancy')
    ax.set_xticks(np.arange(len(speed_values)))
    ax.set_xticklabels([f'{speed:.1f}' for speed in speed_values])

# ---------------------------------------------------------*/
# Reward Calculaction
# ---------------------------------------------------------*/


def plot_rewards(ax, env, belt_speeds, occupancies, title):
    for occupancy in occupancies:
        rewards = []
        for speed in belt_speeds:
            env.belt_speed = speed
            env.belt_occupancy = occupancy
            env.update_accuracy()
            reward = env.calculate_reward()
            rewards.append(reward)

        ax.plot(belt_speeds, rewards, label=f'Occupancy {occupancy*100:.0f}%')

    # ax.set_title(title)
    ax.set_xlabel('Belt Speed')
    ax.set_ylabel('Reward')
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def generate_single_reward_plot(ax, env, belt_speeds):
    reward_factors = (env.reward_factor_accuracy, env.reward_factor_speed)
    accuracies = np.arange(0.6, 1.001, 0.01)

    for speed in belt_speeds:
        rewards = []
        for accuracy in accuracies:
            env.belt_speed = speed
            env.accuracy_belt = [accuracy] * 10  # Mock accuracy values
            reward = env.calculate_reward()
            rewards.append(reward)

        ax.plot(accuracies, rewards, label=f'Speed {speed:.1f}')

    # ax.set_title(f'Reward for Accuracy-Factor={reward_factors[0]:.1f} and Speed-Factor={reward_factors[1]:.1f}')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Reward')
    # ax.legend(loc='lower right')
    ax.grid(True)

    # Set the y-axis scale from -0.5 to 1 with steps of 0.25
    ax.set_ylim(-0.5, 1)
    ax.set_yticks(np.arange(-0.5, 1.25, 0.25))


def generate_reward_heatplot(fig, ax, env):
    accuracy_values = np.linspace(0.6, 1.0, 100)
    speed_values = env.belt_speeds
    reward_matrix = np.zeros((len(accuracy_values), len(speed_values)))

    for i, accuracy in enumerate(accuracy_values):
        for j, speed in enumerate(speed_values):
            env.belt_speed = speed
            env.accuracy_belt = [accuracy] * 10
            reward_matrix[i, j] = env.calculate_reward()

    # Define the colormap with black for rewards below 0 and a more detailed gradient for rewards between 0 and 1
    colors = ['black', 'navy', 'blue', 'dodgerblue', 'cyan', 'lime', 'green', 'yellow', 'gold', 'orange', 'red']
    cmap = ListedColormap(colors)
    # Define boundaries for 10 steps between -1 and 1 with more granularity
    boundaries = [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    c = ax.imshow(reward_matrix, aspect='auto',
                  extent=[-0.5, len(speed_values)-0.5, 0.6, 1.0], origin='lower', cmap=cmap, norm=norm)
    cbar = fig.colorbar(c, ax=ax, ticks=boundaries)
    cbar.set_label('Reward')
    # ax.set_title('Reward based on Speed and Accuracy')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(np.arange(len(speed_values)))
    ax.set_xticklabels([f'{speed:.1f}' for speed in speed_values])

# ---------------------------------------------------------*/
# Run Analysis on Environment
# ---------------------------------------------------------*/


def run_env_analysis(env):
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    axes = axes.flatten()

    env.noise_accuracy = 0              # Set Accuracy-Noise to 0 for analysis
    belt_speeds = env.belt_speeds

    fig.suptitle('Environment Analysis', fontsize=20)
    generate_single_accuracy_plot(axes[0], env, belt_speeds, occupancies)
    env.reset()
    generate_accuracy_heatplot(fig, axes[1], env, belt_speeds)
    env.reset()
    generate_single_reward_plot(axes[2], env, belt_speeds)
    env.reset()
    generate_reward_heatplot(fig, axes[3], env)

    # env.reset()
    # generate_reward_heatplot_occ(fig, axes[4], env)

    plt.tight_layout()
    plt.savefig("./img/plots/combined_analysis.svg", format="svg", bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------*/
# Main
# ---------------------------------------------------------*/
if __name__ == "__main__":
    env = SortingEnvironment()
    run_env_analysis(env)


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
