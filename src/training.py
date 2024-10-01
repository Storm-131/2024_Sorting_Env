#---------------------------------------------------------*\
# Title: Training RL Agent
# Author: TM 05.2024
#---------------------------------------------------------*/

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback
import os


def RL_Trainer(model_type="DQN", env=None, total_timesteps=100_000, tag=""):

    if env is None:
        raise ValueError("Environment must be provided")
    
    env = Monitor(env)
    check_env(env)

    # Create log directory
    tensorboard_log = "./log/tensorboard/"
    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)

    # Choose the model type
    if model_type == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, ent_coef=0.01)
    elif model_type == "DQN":
        model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log)
    elif model_type == "A2C":
        model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, ent_coef=0.01)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Save the model
    model_path = f"./models/{model_type.lower()}_sorting_env_{tag}"
    if os.path.exists(model_path):
        i = 1
        while os.path.exists(f"{model_path}_{i}"):
            i += 1
        model_path = f"{model_path}_{i}"
        
    model.save(model_path)

    # Load the model
    model = model.__class__.load(model_path, env=env)
    
    # Check if model was saved successfully
    if model is not None:
        print(f"Model saved successfully at {model_path}.")
    else:
        print("Failed to save model.")
        
    return model

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\