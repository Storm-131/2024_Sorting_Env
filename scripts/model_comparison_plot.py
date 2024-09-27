# ---------------------------------------------------------*\
# Title: Model Comparison Plot
# Author: TM 2024
# ---------------------------------------------------------*/

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 20})


# Data for plotting
data = {
    'Setup': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'Env': ['Basic', 'Adv', 'Basic', 'Adv', 'Basic', 'Adv', 'Basic', 'Adv'],
    'Random': [11.04, 10.65, -1.24, -2.27, 12.17, 10.98, -1.97, -1.99],
    'RBA': [27.02, 36.71, 12.55, 22.58, 21.07, 30.18, 6.41, 16.28],
    'A2C': [24.96, 34.46, 16.73, 27.27, 24.02, 33.49, 16.73, 24.1],
    'PPO': [26.65, 35.67, 24.85, 33.19, 24.0, 33.77, 18.38, 27.86],
    'DQN': [24.1, 31.35, 20.71, 26.25, 23.33, 31, 18.54, 25.55]
}

df = pd.DataFrame(data)

# Manually defined colors that match the image
blue_color = '#4DB6E2'
green_color = '#A4D494'

colors = [blue_color, green_color]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)

setups = df['Setup'].unique()
envs = df['Env'].unique()
methods = ['Random', 'RBA', 'DQN', 'A2C', 'PPO']

for i, setup in enumerate(setups):
    ax = axes[i // 2, i % 2]
    width = 0.35  # Width of the bars
    x = np.arange(len(methods))
    
    for j, env in enumerate(envs):
        subset = df[(df['Setup'] == setup) & (df['Env'] == env)]
        ax.bar(x + j*width, subset[methods].values.flatten(), width=width, 
               color=colors[j], edgecolor='black', label=f'{env}')
    
    ax.set_title(f'Setup {setup}')
    ax.set_ylabel('Performance')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods)
    if i == 0:
        ax.legend()
    else:
        ax.legend().set_visible(False)

plt.tight_layout()
plt.savefig('../img/figures/model_comparison_plot.pdf')
plt.savefig('../img/figures/model_comparison_plot.svg')
plt.show()



# -------------------------Notes-----------------------------------------------*\
# Receive the values from the Output of the Main.py file and plot them
# -----------------------------------------------------------------------------*\
