# ---------------------------------------------------------*\
# Title: Plotting
# Author: TM 05.2024
# ---------------------------------------------------------*/

import cv2
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re

# ---------------------------------------------------------*/
# Global font increase setting
# ---------------------------------------------------------*/
font_increase = 0  # Adjust this value to increase or decrease the font size globally

# ---------------------------------------------------------*/
# Plot the current state of the sorting environment
# ---------------------------------------------------------*/

def plot_env(material_composition, current_material_belt, current_material_sorting, container_materials, accuracy,
             prev_accuracy, belt_speed, reward_data, belt_occupancy, save=True, show=True, log_dir='./img/log/',
             filename='sorting_system_diagram', sorting_mode=False, title="", format="svg"):

    # Seaborn-Stil
    sns.set_theme(style="whitegrid")
    # Farbschema und Stil
    colors = {'A': 'lightblue', 'B': 'lightgreen', 'Other': 'grey'}

    # Erstellen der Figure und Axes
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(f"Sorting System {title}", fontsize=20 + font_increase, fontweight='bold')

    # Erstellen einzelner Subplots
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=1)
    ax4 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
    ax6 = plt.subplot2grid((2, 4), (1, 1), colspan=2)
    ax7 = plt.subplot2grid((2, 4), (1, 3), colspan=1)

    # ---------------------------------------------------------*/
    # 1) Input Material Composition
    # ---------------------------------------------------------*/
    ax1.set_title('Input', fontweight='bold', fontsize=12 + font_increase)
    ax1.pie(material_composition, labels=['A', 'B'], autopct=lambda p: f'{p:.1f}%', colors=[colors['A'], colors['B']], textprops={'fontsize': 10 + font_increase})
    total = sum(material_composition)
    ax1.text(0.02, 0.98, f'Total: {total}', transform=ax1.transAxes, ha='left', va='top', fontweight='bold', fontsize=10 + font_increase)

    # ---------------------------------------------------------*/
    # 2) Conveyor Belt Material
    # ---------------------------------------------------------*/
    ax2.set_title('Conveyor Belt', fontweight='bold', fontsize=12 + font_increase)
    bars = ax2.bar(['A', 'B'], current_material_belt, color=[colors['A'], colors['B']])
    ax2.text(0.02, 0.98, f'Total: {sum(current_material_belt)}',
             transform=ax2.transAxes, ha='left', va='top', fontweight='bold', fontsize=10 + font_increase)
    ax2.set_ylabel('Quantity', fontsize=10 + font_increase)
    ax2.set_ylim([0, 100])
    # ax2.set_xlabel('Material Type', fontsize=10 + font_increase)
    ax2.tick_params(axis='x', labelsize=10 + font_increase)
    ax2.tick_params(axis='y', labelsize=10 + font_increase)

    for bar, value in zip(bars, current_material_belt):
        ax2.text(bar.get_x() + bar.get_width() / 2, 0, str(value),
                 ha='center', va='bottom', color='black', fontsize=10 + font_increase)

    # ---------------------------------------------------------*/
    # 3) Sorting Machine Material
    # ---------------------------------------------------------*/
    ax3.set_title('Sorting Machine', fontweight='bold', fontsize=12 + font_increase)
    bars = ax3.bar(['A', 'B'], current_material_sorting, color=[colors['A'], colors['B']])
    ax3.text(0.02, 0.98, f'Total: {sum(current_material_sorting)}',
             transform=ax3.transAxes, ha='left', va='top', fontweight='bold', fontsize=10 + font_increase)
    ax3.set_ylim([0, 100])
    # ax3.set_xlabel('Material Type', fontsize=10 + font_increase)
    ax3.tick_params(axis='x', labelsize=10 + font_increase)
    ax3.tick_params(axis='y', labelsize=10 + font_increase)

    accuracy_A = round(prev_accuracy[0], 2)
    accuracy_B = round(prev_accuracy[1], 2)
    ax3.text(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height(),
             f'Accuracy: {accuracy_A}', ha='center', va='bottom', fontsize=10 + font_increase)
    ax3.text(bars[1].get_x() + bars[1].get_width() / 2, bars[1].get_height(),
             f'Accuracy: {accuracy_B}', ha='center', va='bottom', fontsize=10 + font_increase)

    for bar, value in zip(bars, current_material_sorting):
        ax3.text(bar.get_x() + bar.get_width() / 2, 0, str(value),
                 ha='center', va='bottom', color='black', fontsize=10 + font_increase)

    # ---------------------------------------------------------*/
    # 4) Container Contents
    # ---------------------------------------------------------*/
    ax4.set_title('Container Contents', fontweight='bold', fontsize=12 + font_increase)
    colors = {'A': 'lightblue', 'A_False': 'lightgreen', 'B': 'lightgreen', 'B_False': 'lightblue'}
    grouped_keys = [['A', 'A_False'], ['B', 'B_False']]  # Define how the keys are grouped into bars
    ax4.set_xticks(np.arange(len(grouped_keys)))
    ax4.set_xticklabels(['A', 'B'], fontsize=10 + font_increase)  # Label each bar by the primary material key
    ax4.set_ylabel('Quantity', fontsize=10 + font_increase)
    ax4.tick_params(axis='x', labelsize=10 + font_increase)
    ax4.tick_params(axis='y', labelsize=10 + font_increase)

    total_container_contents = sum(container_materials.values())  # Calculate total container contents

    for index, group in enumerate(grouped_keys):
        bottoms = 0  # Reset bottoms for each new bar
        total = 0  # Reset total for each new bar
        for key in group:
            value = container_materials.get(key, 0)
            total += value
            bar = ax4.bar(index, value, bottom=bottoms, color=colors[key], label=key if bottoms == 0 else "")
            ax4.text(index, bottoms, str(value), ha='center', va='bottom', color='black', fontsize=10 + font_increase)
            bottoms += value
        # Calculate the ratio and print it on the plot
        ratio = container_materials.get(group[0], 0) / total * 100  # Calculate the ratio in percentage
        ax4.text(index, total / 2, f"{ratio:.0f}%", ha='center',
                 va='center', color='white', fontweight='bold', fontsize=14 + font_increase)

    # Print total container contents in the left upper corner
    ax4.text(0.02, 0.98, f"Total: {total_container_contents}", ha='left',
             va='top', transform=ax4.transAxes, fontweight='bold', fontsize=10 + font_increase,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # ---------------------------------------------------------*/
    # 5) Next Sorting (Belt Status)
    # ---------------------------------------------------------*/
    ax5.set_title('Next Sorting (Belt Status)', fontweight='bold', fontsize=12 + font_increase)
    light_purple = '#BF40BF'  # RGB
    bars = ax5.bar(['A (acc)', 'B (acc)', 'Quantity', 'Speed'], accuracy + [belt_occupancy, belt_speed],
                   color=[colors['A'], colors['B'], 'red', light_purple])
    ax5.axhline(y=1, color='red')  # Zeichnet eine vertikale rote Linie bei X = 1
    ax5.set_ylim([0, 1])
    ax5.set_yticks(np.arange(0, 1.1, 0.2))
    ax5.set_ylabel('Percentage (/100)', fontsize=10 + font_increase)
    ax5.tick_params(axis='x', labelsize=10 + font_increase)
    ax5.tick_params(axis='y', labelsize=10 + font_increase)

    # Add values as text at the bottom of the corresponding bars
    values = accuracy + [belt_occupancy, belt_speed]
    for i, bar in enumerate(bars):
        formatted_value = f"{values[i]:.2f}"
        ax5.text(bar.get_x() + bar.get_width() / 2, 0, formatted_value, ha='center', va='bottom', fontsize=10 + font_increase)

    if sorting_mode:
        mode_names = ['Basic', 'Positive', 'Negative']
        current_mode = mode_names[reward_data['Modes'][-1]] if reward_data['Modes'] else 'Unknown'
        ax5.text(0.98, 0.98, f'Mode: {current_mode}', transform=ax5.transAxes, ha='right', va='top', fontweight='bold',
                 fontsize=10 + font_increase, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # ---------------------------------------------------------*/
    # 6) Current Reward Metrics
    # ---------------------------------------------------------*/
    ax6.set_title('Current Reward Metrics', fontweight='bold', fontsize=12 + font_increase)
    x_axe = range(len(reward_data['Reward']))
    ax6.plot(x_axe, reward_data['Reward'], label='Reward', color='darkgreen', linewidth=3)
    ax6.plot(x_axe, reward_data['Speed'], label='Speed', color='purple', linewidth=2, alpha=0.6)
    ax6.plot(x_axe, reward_data['Accuracy'], label='Accuracy', color='blue', linewidth=2, alpha=0.6)
    ax6.plot(x_axe, reward_data['Occupancy'], label='Occupancy', color='red', linewidth=2, alpha=0.6)
    if sorting_mode:    # Modes divided by 2 to better fit the plot
        ax6.step(x_axe, np.array(reward_data['Modes']) / 4, label='Modes',
                 color='orange', linewidth=2, alpha=0.6, where='post')
    ax6.legend(loc='lower left', fontsize=8 + font_increase)
    ax6.set_xlim([0, None])
    ax6.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%d' % (x)))  # Setze x-Achse auf Ganzzahlen
    ax6.set_ylim([-0.2, 1.2])  # Adjusted ylim to accommodate mode values
    ax6.yaxis.set_ticks(np.arange(-0.2, 1.201, 0.2))  # Set step size to 0.2
    ax6.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax6.set_xlabel('Timesteps', fontsize=10 + font_increase)
    ax6.tick_params(axis='x', labelsize=10 + font_increase)
    ax6.tick_params(axis='y', labelsize=10 + font_increase)

    # Calculate the mean speed and display it in the top right corner of the plot
    mean_speed = np.mean(reward_data['Speed'])

    # Fügen Sie den Durchschnittswert als Text in die obere rechte Ecke des Plots ein
    ax6.text(0.98, 0.98, f'Mean Speed: {mean_speed:.0%}',
             horizontalalignment='right', verticalalignment='top', transform=ax6.transAxes, fontweight='bold',
             fontsize=10 + font_increase, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    # ---------------------------------------------------------*/
    # 7) Cumulative Reward Metrics (Line Plot)
    # ---------------------------------------------------------*/
    ax7.set_title('Cumulative Reward Metrics', fontweight='bold', fontsize=12 + font_increase)
    total_reward = np.cumsum(reward_data['Reward'])
    ax7.plot(x_axe, total_reward, color='darkgreen')
    ax7.set_xlim([0, None])
    ax7.set_ylim([0, None])
    ax7.set_xlabel('Timesteps', fontsize=10 + font_increase)
    # Set x-axis limits and ticks
    ax7.set_xlim([0, 50])
    ax7.set_xticks(np.arange(0, 51, 5))  # Set x-axis ticks from 0 to 50 with steps of 5
    ax7.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax7.tick_params(axis='x', labelsize=10 + font_increase)
    ax7.tick_params(axis='y', labelsize=10 + font_increase)
    # Select the last value of the cumulative reward to display as the total reward
    final_total_reward = round(total_reward[-1] if total_reward.size > 0 else 0, 2)
    ax7.text(ax7.get_ylim()[1]*0.02, ax7.get_ylim()[1]*0.98, f'Total: {final_total_reward}', va='top',
             ha='left', fontweight='bold', fontsize=10 + font_increase, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    if save:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fig.tight_layout()
        save_plot(fig, log_dir, filename, extension=format)

    if show:
        plt.tight_layout()
        plt.show()

    plt.close()

# ---------------------------------------------------------*/
# Save the plot to a file
# ---------------------------------------------------------*/


def save_plot(fig, dir, base_filename, extension, dpi=300):
    """Saves the plot to a file with a unique filename in the specified directory."""

    os.makedirs(dir, exist_ok=True)  # Ensure the directory exists

    # Generate unique filename
    i = 0
    filename = f"{base_filename}_{i}.{extension}"
    while os.path.exists(os.path.join(dir, filename)):
        i += 1
        filename = f"{base_filename}_{i}.{extension}"

    # Save the figure
    if extension == 'svg':
        fig.savefig(os.path.join(dir, filename), format=extension)
    else:
        fig.savefig(os.path.join(dir, filename), format=extension, dpi=dpi, bbox_inches='tight')

# ---------------------------------------------------------*/
# Create a Video from a folder of images
# ---------------------------------------------------------*/


def create_video(folder_path, output_path, display_duration=1):
    """ Creates a video from a folder of images.
    - folder_path: The path to the folder containing the images.
    - output_path: The path to save the output video.
    - display_duration: The duration (in seconds) to display each image.
    """
    sample_img = cv2.imread(os.path.join(folder_path, os.listdir(folder_path)[0]))
    height, width, layers = sample_img.shape
    size = (width, height)

    # Frame-Rate basierend auf der gewünschten Anzeigedauer jedes Bildes berechnen
    frame_rate = 1 / display_duration

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)

    # Funktion zum Extrahieren der Nummern aus dem Dateinamen für die korrekte Sortierung
    def sort_key(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    # Dateien numerisch sortieren und Video schreiben
    for filename in sorted(os.listdir(folder_path), key=sort_key):
        if filename.endswith('.png'):
            # print("Added ", filename)
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img = cv2.resize(img, size)  # Resize the image to the target size
            out.write(img)

    print("Video created. 🎥🍿")
    out.release()

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\

