# ---------------------------------------------------------*\
# Title: Simulations
# Author: TM 05.2024
# ---------------------------------------------------------*/

import os
import shutil
from tqdm import tqdm
from utils.plotting import create_video
import time

# ---------------------------------------------------------*/
# Interactive Simulation
# ---------------------------------------------------------*/


def interactive_simulation(env=None, steps=50):
    if env is None:
        raise ValueError("Environment must be provided")

    print("Starting interactive simulation...")
    time.sleep(2)

    env.print_possible_actions()

    obs, _ = env.reset()
    print("\nInitial Observation:", obs)

    for i in range(steps):
        action = int(input(f"Step {i+1} - Choose an action (0-14): "))
        if action < 0 or action >= env.action_space.n:
            print("Invalid action. Please choose a valid action between 0 and 14.")
            continue

        obs, reward, done, _, _ = env.step(action)
        print(f"Step {i+1} - Action: {action}, Observation: {obs}, Reward: {reward}")
        env.render(save=False, log_dir="./img/", filename=f'interactive_step_{i+1}', title=f'Step {i+1}', format='png')

        if done:
            print("Episode finished.")
            obs, _ = env.reset()
            break

# ---------------------------------------------------------*/
# Simulations with Video-Recordings
# ---------------------------------------------------------*/


def env_simulation_video(env=None, tag="", title="Environment Simulation", steps=50, dir="./img/"):

    if env is None:
        raise ValueError("Environment must be provided")

    temp_dir = f"{dir}temp/"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    obs, _ = env.reset()
    print("Initial Observation:", obs)

    for i in tqdm(range(steps), disable=False):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        # print(f"Step {i+1} - Action: {action}, Observation: {obs}, Reward: {reward}")
        env.render(save=True, show=False, log_dir=temp_dir, filename=f'{tag}_env_simulation', title=title, format='png')
        if done:
            env.render(save=True, show=False, log_dir=temp_dir,
                       filename=f'{tag}_env_simulation', title=title, format='png')
            obs, _ = env.reset()

    print("Creating video...")
    create_video(folder_path=temp_dir, output_path=f'{dir}{title}_{tag}.mp4')

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def model_simulation_video(model, env=None, tag="", title="Model Simulation", steps=50, dir="./img/"):

    if env is None:
        raise ValueError("Environment must be provided")

    temp_dir = f"{dir}temp/"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    obs, _ = env.reset()
    print("Initial Observation:", obs)

    for i in tqdm(range(steps), disable=False):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        # print(f"Step {i+1} - Action: {action}, Observation: {obs}, Reward: {reward}")
        env.render(save=True, show=False, log_dir=temp_dir, filename=f'{tag}_env_simulation', title=title, format='png')
        if done:
            env.render(save=True, show=False, log_dir=temp_dir,
                       filename=f'{tag}_{model.__class__.__name__}_simulation', title=title, format='png')
            obs, _ = env.reset()

    print("Creating video...")
    create_video(folder_path=temp_dir, output_path=f'{dir}{title}_{tag}.mp4')

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

# -------------------------Notes-----------------------------------------------*\

# -----------------------------------------------------------------------------*\
