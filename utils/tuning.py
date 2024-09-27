# ---------------------------------------------------------*\
# Title: Tuning class (for simple env)
# Author:
# ---------------------------------------------------------*/

import os
import csv
import numpy as np
import optuna
from stable_baselines3 import PPO, DQN, A2C
from itertools import product
from tqdm import tqdm

from src.rule_based_agent import Rule_Based_Agent_simple
from src.env_1_simple import SortingEnvironment

# ---------------------------------------------------------*/
# Tuning Class
# ---------------------------------------------------------*/


class Tuning:
    def __init__(self, models, tag="", n_trials=100):
        self.models = models
        self.results = {}
        self.tag = tag
        self.seed = 99
        self.n_trials = n_trials

        # Initialize the CSV file with a header
        with open(f"./log/tuning_results_{self.tag}.csv", "w", newline='') as f:
            writer = csv.writer(f)
            header = ['Group', 'Model Type', 'Input', 'Noise', 'Action Penalty', 'Learning Rate', 'Entropy Coef', 'Gamma', 'N Eval Episodes',
                      'Total Timesteps', 'Train Steps', 'Test Steps', 'Mean Reward', 'Standard Deviation Reward', 'Total Reward Seed 42']
            writer.writerow(header)

        # Tuning Parameters
        # Input Type r=random, s3=simple_saisonal, s9=complex_saisonal
        self.INPUT = ["r", "s3", "s9"]
        self.NOISE = [0.0, 0.1, 0.2, 0.3]                               # Noise Range (0.0 - 1.0)
        self.ACTION_PENALTY = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]            # Action Penalty for Taking Too Many Actions
        self.LEARNING_RATE = [0.0001, 0.0003, 0.0005, 0.0007, 0.001]    # Learning Rate for the Model
        self.ENTROPY_COEF = [0.01, 0.03, 0.05, 0.07, 0.10]              # Entropy Coefficient
        self.GAMMA = [0.90, 0.92, 0.94, 0.96, 0.98]                     # Discount Factor
        self.N_EVAL_EPISODES = range(5, 21, 5)                          # Number of Evaluation Episodes
        self.TOTAL_TIMESTEPS = range(100_000, 200_001, 50_000)          # Total Training Steps (Budget)
        self.TRAIN_TIMESTEPS = range(50, 251, 50)                       # Steps per Episode (Training)

        self.param_grid = list(product(self.INPUT, self.NOISE, self.ACTION_PENALTY, self.LEARNING_RATE,
                               self.ENTROPY_COEF, self.GAMMA, self.N_EVAL_EPISODES, self.TOTAL_TIMESTEPS, self.TRAIN_TIMESTEPS))

        self.rba_param_grid = list(product(self.INPUT, self.NOISE, self.ACTION_PENALTY))

    def run_tuning(self, dir="./img/"):
        """Run benchmark for all models and parameter combinations."""
        # Handle RBA separately
        if "RBA" in self.models:
            for rba_params in self.rba_param_grid:
                train_env = self.make_env_rba(rba_params, training=True)
                eval_env = self.make_env_rba(rba_params, training=False)
                mean_reward, std_reward, total_reward_seed_42 = self.train_and_evaluate_rule_based_agent(
                    train_env, eval_env, n_eval_episodes=50)
                self.results[("RBA", rba_params)] = {
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'total_reward_seed_42': total_reward_seed_42
                }
                print(f"Added: RBA, Params: {rba_params}, Mean Reward: {mean_reward}, Std Reward: {std_reward},\
                    Total Reward Seed 42: {total_reward_seed_42}")
                self.append_to_csv(0, "RBA", rba_params, mean_reward, std_reward, total_reward_seed_42)
            self.models.remove("RBA")  # Remove RBA from the models

        if self.models != []:
            # Initialize the progress bar
            progress_bar = tqdm(total=len(self.param_grid) * len(self.models))  # Initialize the progress bar
            group = 1

            for params in self.param_grid:
                train_env = self.make_env(params, training=True)
                eval_env = self.make_env(params, training=False)

                for model_type in self.models:

                    result = self.run_single_experiment(params, model_type, train_env, eval_env)
                    model_type, params, mean_reward, std_reward, total_reward_seed_42 = result
                    self.results[(model_type, params)] = {
                        'mean_reward': mean_reward,
                        'std_reward': std_reward,
                        'total_reward_seed_42': total_reward_seed_42
                    }
                    print(
                        f"Added: {model_type}, Params: {params}, Mean Reward: {mean_reward}, Std Reward: {std_reward}, Total Reward Seed 42: {total_reward_seed_42}")

                    # Append to CSV
                    self.append_to_csv(group, model_type, params, mean_reward, std_reward, total_reward_seed_42)

                    progress_bar.update(1)  # Update the progress bar

                group += 1  # Increment the group counter

            progress_bar.close()

    def run_single_experiment(self, params, model_type, train_env, eval_env):
        input_type, noise, action_penalty, lr, ent_coef, gamma, n_eval_episodes, total_timesteps, train_timesteps = params

        train_env.reset(seed=self.seed)
        model, _ = self.train_model(model_type, train_env, lr, ent_coef, gamma, total_timesteps)
        mean_reward, std_reward, total_reward_seed_42 = self.evaluate_model(model, eval_env, n_eval_episodes)

        return model_type, params, mean_reward, std_reward, total_reward_seed_42

    def make_env(self, params, training):
        input_type, noise, action_penalty, lr, ent_coef, gamma, n_eval_episodes, total_timesteps, train_timesteps = params
        max_steps = train_timesteps if training else 50
        print(f"Creating {'Training' if training else 'Evaluation'} Environment with input_type={input_type}, noise={noise}, action_penalty={action_penalty}, max_steps={max_steps}, seed={self.seed}")
        env = SortingEnvironment(max_steps=max_steps, seed=self.seed, noise_lv=noise,
                                 input=input_type, action_penalty=action_penalty)
        return env

    def make_env_rba(self, params, training):
        input_type, noise, action_penalty = params
        max_steps = 50
        print(f"Creating {'Training' if training else 'Evaluation'} Environment with input_type={input_type}, noise={noise}, action_penalty={action_penalty}, max_steps={max_steps}, seed={self.seed}")
        env = SortingEnvironment(max_steps=max_steps, seed=self.seed, noise_lv=noise,
                                 input=input_type, action_penalty=action_penalty)
        return env

    def get_model(self, model_type, env, lr, ent_coef, gamma, tensorboard_log):
        """Return a model based on the model type and parameters."""
        if model_type == "PPO":
            return PPO("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed, ent_coef=ent_coef,
                       learning_rate=lr, gamma=gamma)
        elif model_type == "DQN":
            return DQN("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed, learning_rate=lr, gamma=gamma)
        elif model_type == "A2C":
            return A2C("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed, ent_coef=ent_coef,
                       learning_rate=lr, gamma=gamma)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train_model(self, model_type, env, lr, ent_coef, gamma, total_timesteps):
        """Train a model on the stochastic Sorting Environment ("no seed")"""
        tensorboard_log = f"./log/tensorboard/{model_type}/"
        os.makedirs(tensorboard_log, exist_ok=True)

        model = self.get_model(model_type, env, lr, ent_coef, gamma, tensorboard_log=None)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        return model, env

    def evaluate_model(self, model, eval_env, n_eval_episodes):
        """Evaluate a model over n episodes and give back the mean and std of the rewards.
        This uses a seed for the evaluation episodes."""

        rewards = []
        eval_env.reset(seed=self.seed)

        for episode in range(n_eval_episodes):
            obs, _ = eval_env.reset(seed=42+episode)

            total_reward = 0
            done = False

            while not done:
                if model is None:
                    action = eval_env.action_space.sample()
                else:
                    action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, _, _ = eval_env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        mean_reward = round(np.mean(rewards), 2)
        std_reward = round(np.std(rewards), 2)
        total_reward_seed_42 = round(rewards[0], 2)

        return mean_reward, std_reward, total_reward_seed_42

    def train_and_evaluate_rule_based_agent(self, train_env, eval_env, n_eval_episodes):
        """Train and evaluate the Rule-Based Agent over n episodes and give back the mean and std of the rewards.
        This uses a seed for the evaluation episodes."""
        train_env.reset(seed=self.seed)
        # Train the Rule-Based Agent
        agent = Rule_Based_Agent_simple(train_env)

        eval_env.reset(seed=self.seed)

        # Evaluate the Rule-Based Agent
        rewards = []

        for episode in range(n_eval_episodes):
            obs, _ = eval_env.reset(seed=42+episode)

            total_reward = 0
            done = False

            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        mean_reward = round(np.mean(rewards), 2)
        std_reward = round(np.std(rewards), 2)
        total_reward_seed_42 = round(rewards[0], 2)

        return mean_reward, std_reward, total_reward_seed_42

    def append_to_csv(self, group, model_type, params, mean_reward, std_reward, total_reward_seed_42):
        """Append the results to the CSV file immediately."""
        if model_type == "RBA":
            input_type, noise, action_penalty = params
            row = [group, model_type, input_type, noise, action_penalty, None, None, None, None,
                   None, None, 50, mean_reward, std_reward, total_reward_seed_42]
        else:
            input_type, noise, action_penalty, lr, ent_coef, gamma, n_eval_episodes, total_timesteps, train_timesteps = params
            row = [group, model_type, input_type, noise, action_penalty, lr, ent_coef, gamma, n_eval_episodes,
                   total_timesteps, train_timesteps, 50, mean_reward, std_reward, total_reward_seed_42]
        with open(f"./log/tuning_results_{self.tag}.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


# ---------------------------------------------------------*/
# Optuna - Tuning Class
# ---------------------------------------------------------*/

class Tuning_Optuna:
    def __init__(self, models, tag="", n_trials=100):
        self.models = models
        self.results = {}
        self.tag = tag
        self.seed = 99
        self.n_trials = n_trials

        # Tuning Parameters
        self.INPUT = ["r", "s3", "s9"]
        self.NOISE = [0.0, 0.1, 0.2, 0.3]                               # Noise Range (0.0 - 1.0)
        self.ACTION_PENALTY = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]            # Action Penalty for Taking Too Many Actions
        self.LEARNING_RATE = [0.0001, 0.0003, 0.0005, 0.0007, 0.001]    # Learning Rate for the Model
        self.ENTROPY_COEF = [0.01, 0.03, 0.05, 0.07, 0.10]              # Entropy Coefficient
        self.GAMMA = [0.90, 0.92, 0.94, 0.96, 0.98]                     # Discount Factor
        self.N_EVAL_EPISODES = range(5, 21, 5)                          # Number of Evaluation Episodes
        self.TOTAL_TIMESTEPS = range(50_000, 150_000, 50_000)           # Total Training Steps (Budget)
        self.TRAIN_TIMESTEPS = range(50, 251, 50)                       # Steps per Episode (Training)

        self.param_grid = list(product(self.INPUT, self.NOISE, self.ACTION_PENALTY, self.LEARNING_RATE,
                                       self.ENTROPY_COEF, self.GAMMA, self.N_EVAL_EPISODES, self.TOTAL_TIMESTEPS, self.TRAIN_TIMESTEPS))

        self.rba_param_grid = list(product(self.INPUT, self.NOISE, self.ACTION_PENALTY))

        # Initialize the CSV file with a header
        with open(f"./log/tuning_results_{self.tag}.csv", "w", newline='') as f:
            writer = csv.writer(f)
            header = ['Group', 'Model Type', 'Input', 'Noise', 'Action Penalty', 'Learning Rate', 'Entropy Coef', 'Gamma', 'N Eval Episodes',
                      'Total Timesteps', 'Train Steps', 'Test Steps', 'Mean Reward', 'Standard Deviation Reward', 'Total Reward Seed 42']
            writer.writerow(header)

    def objective(self, trial):
        input_type = trial.suggest_categorical('input_type', self.INPUT)
        noise = trial.suggest_categorical('noise', self.NOISE)
        action_penalty = trial.suggest_categorical('action_penalty', self.ACTION_PENALTY)
        lr = trial.suggest_categorical('learning_rate', self.LEARNING_RATE)
        ent_coef = trial.suggest_categorical('entropy_coef', self.ENTROPY_COEF)
        gamma = trial.suggest_categorical('gamma', self.GAMMA)
        n_eval_episodes = trial.suggest_categorical('n_eval_episodes', self.N_EVAL_EPISODES)
        total_timesteps = trial.suggest_categorical('total_timesteps', self.TOTAL_TIMESTEPS)
        train_timesteps = trial.suggest_categorical('train_timesteps', self.TRAIN_TIMESTEPS)

        params = (input_type, noise, action_penalty, lr, ent_coef, gamma,
                  n_eval_episodes, total_timesteps, train_timesteps)

        best_mean_reward = -np.inf
        best_model_type = None

        for model_type in self.models:
            train_env = self.make_env(params, training=True)
            eval_env = self.make_env(params, training=False)

            result = self.run_single_experiment(params, model_type, train_env, eval_env)
            model_type, params, mean_reward, std_reward, total_reward_seed_42 = result

            self.append_to_csv(trial.number, model_type, params, mean_reward, std_reward, total_reward_seed_42)

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_model_type = model_type

        return best_mean_reward

    def run_tuning(self, dir="./img/"):
        """Run benchmark for all models and parameter combinations."""
        # Handle RBA separately
        if "RBA" in self.models:
            for rba_params in self.rba_param_grid:
                train_env = self.make_env_rba(rba_params, training=True)
                eval_env = self.make_env_rba(rba_params, training=False)
                mean_reward, std_reward, total_reward_seed_42 = self.train_and_evaluate_rule_based_agent(
                    train_env, eval_env, n_eval_episodes=50)
                self.results[("RBA", rba_params)] = {
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'total_reward_seed_42': total_reward_seed_42
                }
                print(f"Added: RBA, Params: {rba_params}, Mean Reward: {mean_reward}, Std Reward: {std_reward},\
                    Total Reward Seed 42: {total_reward_seed_42}")
                self.append_to_csv(0, "RBA", rba_params, mean_reward, std_reward, total_reward_seed_42)
            self.models.remove("RBA")  # Remove RBA from the models

        if self.models != []:
            study = optuna.create_study(direction='maximize')
            study.optimize(self.objective, n_trials=self.n_trials)

            # Save best trial parameters
            best_trial = study.best_trial
            best_params = best_trial.params
            print(f"Best Trial: {best_params}")

    def run_single_experiment(self, params, model_type, train_env, eval_env):
        input_type, noise, action_penalty, lr, ent_coef, gamma, n_eval_episodes, total_timesteps, train_timesteps = params

        train_env.reset(seed=self.seed)
        model, _ = self.train_model(model_type, train_env, lr, ent_coef, gamma, total_timesteps)
        mean_reward, std_reward, total_reward_seed_42 = self.evaluate_model(model, eval_env, n_eval_episodes)

        return model_type, params, mean_reward, std_reward, total_reward_seed_42

    def make_env(self, params, training):
        input_type, noise, action_penalty, lr, ent_coef, gamma, n_eval_episodes, total_timesteps, train_timesteps = params
        max_steps = train_timesteps if training else 50
        print(f"Creating {'Training' if training else 'Evaluation'} Environment with input_type={input_type}, noise={noise}, action_penalty={action_penalty}, max_steps={max_steps}, seed={self.seed}")
        env = SortingEnvironment(max_steps=max_steps, seed=self.seed, noise_lv=noise,
                                 input=input_type, action_penalty=action_penalty)
        return env

    def make_env_rba(self, params, training):
        input_type, noise, action_penalty = params
        max_steps = 50
        print(f"Creating {'Training' if training else 'Evaluation'} Environment with input_type={input_type}, noise={noise}, action_penalty={action_penalty}, max_steps={max_steps}, seed={self.seed}")
        env = SortingEnvironment(max_steps=max_steps, seed=self.seed, noise_lv=noise,
                                 input=input_type, action_penalty=action_penalty)
        return env

    def get_model(self, model_type, env, lr, ent_coef, gamma, tensorboard_log):
        """Return a model based on the model type and parameters."""
        if model_type == "PPO":
            return PPO("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed, ent_coef=ent_coef,
                       learning_rate=lr, gamma=gamma)
        elif model_type == "DQN":
            return DQN("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed, learning_rate=lr, gamma=gamma)
        elif model_type == "A2C":
            return A2C("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed, ent_coef=ent_coef,
                       learning_rate=lr, gamma=gamma)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train_model(self, model_type, env, lr, ent_coef, gamma, total_timesteps):
        """Train a model on the stochastic Sorting Environment ("no seed")"""
        tensorboard_log = f"./log/tensorboard/{model_type}/"
        os.makedirs(tensorboard_log, exist_ok=True)

        model = self.get_model(model_type, env, lr, ent_coef, gamma, tensorboard_log=None)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        return model, env

    def evaluate_model(self, model, eval_env, n_eval_episodes):
        """Evaluate a model over n episodes and give back the mean and std of the rewards.
        This uses a seed for the evaluation episodes."""

        rewards = []
        eval_env.reset(seed=self.seed)

        for episode in range(n_eval_episodes):
            obs, _ = eval_env.reset(seed=42+episode)

            total_reward = 0
            done = False

            while not done:
                if model is None:
                    action = eval_env.action_space.sample()
                else:
                    action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, _, _ = eval_env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        mean_reward = round(np.mean(rewards), 2)
        std_reward = round(np.std(rewards), 2)
        total_reward_seed_42 = round(rewards[0], 2)

        return mean_reward, std_reward, total_reward_seed_42

    def train_and_evaluate_rule_based_agent(self, train_env, eval_env, n_eval_episodes):
        """Train and evaluate the Rule-Based Agent over n episodes and give back the mean and std of the rewards.
        This uses a seed for the evaluation episodes."""
        train_env.reset(seed=self.seed)
        # Train the Rule-Based Agent
        agent = Rule_Based_Agent_simple(train_env)

        eval_env.reset(seed=self.seed)

        # Evaluate the Rule-Based Agent
        rewards = []

        for episode in range(n_eval_episodes):
            obs, _ = eval_env.reset(seed=42+episode)

            total_reward = 0
            done = False

            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        mean_reward = round(np.mean(rewards), 2)
        std_reward = round(np.std(rewards), 2)
        total_reward_seed_42 = round(rewards[0], 2)

        return mean_reward, std_reward, total_reward_seed_42

    def append_to_csv(self, group, model_type, params, mean_reward, std_reward, total_reward_seed_42):
        """Append the results to the CSV file immediately."""
        if model_type == "RBA":
            input_type, noise, action_penalty = params
            row = [group, model_type, input_type, noise, action_penalty, None, None, None, None,
                   None, None, 50, mean_reward, std_reward, total_reward_seed_42]
        else:
            input_type, noise, action_penalty, lr, ent_coef, gamma, n_eval_episodes, total_timesteps, train_timesteps = params
            row = [group, model_type, input_type, noise, action_penalty, lr, ent_coef, gamma, n_eval_episodes,
                   total_timesteps, train_timesteps, 50, mean_reward, std_reward, total_reward_seed_42]
        with open(f"./log/tuning_results_{self.tag}.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
