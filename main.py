# ---------------------------------------------------------*\
# Title: Sorting System - Gym Environment (Main)
# Author: TM 2024
# ---------------------------------------------------------*/

from stable_baselines3 import DQN, PPO, A2C
from utils.benchmark import RLBenchmark

from src.testing import test_env, test_model
from src.training import RL_Trainer
from src.env_1_simple import SortingEnvironment
from src.env_2_adv import SortingEnvironmentAdv
from src.rule_based_agent import Rule_Based_Agent_simple, Rule_Based_Agent_adv

from utils.plot_env_stats import run_env_analysis
from utils.tuning import Tuning
from utils.simulation import interactive_simulation, env_simulation_video, model_simulation_video

# ---------------------------------------------------------*/
# Parameters
# ---------------------------------------------------------*/

# 1. Select Mode
# ---------------------------------------------------------*/
TEST = 0            # Test a random run
TRAIN = 0           # Train a new model
BENCHMARK = 0       # Benchmarking multiple models
SMALL_CHECK = 0     # Small Check for Testing
LOAD = 0            # Load a pre-trained model

RULE_BASED = 0      # Rule Based Agent
INTERACTIVE = 0     # For Interactive Mode (Manual Control)
VIDEO = 0           # Record Video, for Test and Load Mode
ENV_ANALYSIS = 0    # Analyse Environment
TUNING = 0          # Tuning

if SMALL_CHECK:
    TEST = 1
    TRAIN = 1
    RULE_BASED = 1
    ENV_ANALYSIS = 1

# ---------------------------------------------------------*/
if TRAIN:
    MODELS = ["DQN"]
else:
    MODELS = ["RBA", "A2C", "PPO", "DQN"]

TAG_ADD = "B_Base"        # Additional Tag for spedific runs

COMPLEX = 0             # Complex Environment (1) or Simple Environment (0)
INPUT = "r"             # Input Type r=random, s3=simple_saisonal, s9=complex_saisonal
THRESHOLD = 0.7         # Threshold for Accuracy
NOISE = 0               # Noise Range (0.0 - 1.0)

if INPUT == "r":
    ACTION_PENALTY = 0      # Action Penalty for Taking Too Many Actions
else:
    ACTION_PENALTY = 0.5    # Action Penalty for Taking Too Many Actions

TIMESTEPS = 100_000     # Total Training Steps (Budget)
STEPS_TRAIN = 250       # Steps per Episode (Training)
STEPS_TEST = 50         # Steps per Episode (Testing)
SEED = 42               # Random Seed for Reproducibility

SAVE = 1                 # Save Images
DIR = "./img/figures/"   # Directory for Image-Logging

# ---------------------------------------------------------*/
# Run Environment
# ---------------------------------------------------------*/


def run_env(TEST=TEST, TRAIN=TRAIN, BENCHMARK=BENCHMARK, RULE_BASED=RULE_BASED, SMALL_CHECK=SMALL_CHECK, LOAD=LOAD,
            INTERACTIVE=INTERACTIVE, VIDEO=VIDEO, ENV_ANALYSIS=ENV_ANALYSIS, TUNING=TUNING, MODELS=MODELS,
            TAG_ADD=TAG_ADD, COMPLEX=COMPLEX, INPUT=INPUT, THRESHOLD=THRESHOLD, NOISE=NOISE,
            ACTION_PENALTY=ACTION_PENALTY, TIMESTEPS=TIMESTEPS, STEPS_TRAIN=STEPS_TRAIN, STEPS_TEST=STEPS_TEST,
            SEED=SEED, SAVE=SAVE, DIR=DIR):

    def create_environment(max_steps=STEPS_TEST, seed=None):
        """Function to create a new environment based on parameters"""
        print(f"Creating environment with: complexity={COMPLEX}, steps={TIMESTEPS}, input_type={INPUT},\
                noise_level={NOISE}, seed={seed}")
        if COMPLEX:
            return SortingEnvironmentAdv(max_steps=max_steps, input=INPUT, action_penalty=ACTION_PENALTY,
                                         noise_lv=NOISE, seed=seed, threshold=THRESHOLD)
        else:
            return SortingEnvironment(max_steps=max_steps, input=INPUT, action_penalty=ACTION_PENALTY,
                                      noise_lv=NOISE, seed=seed, threshold=THRESHOLD)

    if SMALL_CHECK:
        TEST = 1
        TRAIN = 1
        RULE_BASED = 1
        ENV_ANALYSIS = 1

    # Environment Setup (automatic)
    if COMPLEX:
        TAG = f"{TAG_ADD}_adv"
        agent_model = Rule_Based_Agent_adv
    else:
        TAG = f"{TAG_ADD}_base"
        agent_model = Rule_Based_Agent_simple

    if ENV_ANALYSIS:
        env = create_environment()
        run_env_analysis(env)

    if TEST or BENCHMARK:
        env = create_environment(seed=SEED)
        if VIDEO:
            env_simulation_video(env=env, tag=TAG, steps=STEPS_TEST)
        else:
            test_env(env=env, tag=TAG, save=SAVE, title=f"(Random Run, {TAG})", steps=STEPS_TEST, dir=DIR, seed=42)

    if RULE_BASED:
        env = create_environment()
        agent = agent_model(env)
        agent.run_analysis()
        env = create_environment(seed=SEED)
        test_model(agent, env=env, tag=TAG, save=SAVE, title=f"(Rule-Based Agent {TAG})", steps=STEPS_TEST, dir=DIR)

    if TRAIN:
        train_env = create_environment(max_steps=STEPS_TRAIN)
        model = RL_Trainer(model_type=MODELS[0], env=train_env, total_timesteps=TIMESTEPS, tag=TAG)
        env = create_environment(seed=SEED)
        test_model(model, env=env, tag=TAG, save=SAVE,
                   title=f"(Trained Run, {MODELS[0]} {TAG})", steps=STEPS_TEST, dir=DIR)

    if BENCHMARK:
        train_env = create_environment(max_steps=STEPS_TRAIN)
        eval_env = create_environment(max_steps=STEPS_TEST, seed=SEED)
        benchmark = RLBenchmark(models=MODELS, total_timesteps=TIMESTEPS, n_eval_episodes=10, train_env=train_env,
                                eval_env=eval_env, tag=TAG, agent_model=agent_model)
        benchmark.run_benchmark(dir=DIR)
        LOAD = 1

    if LOAD:

        for modelname in MODELS:
            if "DQN" in modelname:
                env = create_environment(seed=SEED)
                model = DQN.load(f"models/{modelname.lower()}_sorting_env_{TAG}")
                test_model(model, env=env, tag=TAG, save=SAVE, title=f"({modelname} {TAG})", steps=STEPS_TEST, dir=DIR)
            elif "PPO" in modelname:
                env = create_environment(seed=SEED)
                model = PPO.load(f"models/{modelname.lower()}_sorting_env_{TAG}")
                test_model(model, env=env, tag=TAG, save=SAVE, title=f"({modelname} {TAG})", steps=STEPS_TEST, dir=DIR)
            elif "A2C" in modelname:
                env = create_environment(seed=SEED)
                model = A2C.load(f"models/{modelname.lower()}_sorting_env_{TAG}")
                test_model(model, env=env, tag=TAG, save=SAVE, title=f"({modelname} {TAG})", steps=STEPS_TEST, dir=DIR)
            elif "RBA" in modelname:
                train_env = create_environment(max_steps=STEPS_TRAIN, seed=100)
                agent = agent_model(train_env)
                env = create_environment(seed=SEED)
                test_model(agent, env=env, tag=TAG, save=SAVE,
                           title=f"(Rule-Based Agent {TAG})", steps=STEPS_TEST, dir=DIR)

            else:
                raise ValueError(f"Unsupported model type: {modelname}")

        if VIDEO:
            model_simulation_video(model, env=env, tag=TAG, title=f"Model Simulation ({modelname})", steps=STEPS_TEST)

    if INTERACTIVE:
        env = create_environment()
        interactive_simulation(env=env)

    if TUNING:
        models = ["RBA", "A2C", "PPO", "DQN"]  # List of models to tune
        tag = "experiment_1"  # Tag for this run

        tuner = Tuning(models=models, tag=tag)
        tuner.run_tuning()

        print("Tuning completed. Results saved. 🎣")


# ---------------------------------------------------------*/
# Main Function
# ---------------------------------------------------------*/
if __name__ == "__main__":

    # Environment Analysis
    run_env(ENV_ANALYSIS=1)

    # A: Random Input
    run_env(BENCHMARK=1, TAG_ADD="A", COMPLEX=0, INPUT="r", NOISE=0, ACTION_PENALTY=0)
    run_env(BENCHMARK=1, TAG_ADD="A", COMPLEX=1, INPUT="r", NOISE=0, ACTION_PENALTY=0)

    # B: Seasonal Input
    run_env(BENCHMARK=1, TAG_ADD="B", COMPLEX=0, INPUT="s9", NOISE=0, ACTION_PENALTY=0.5)
    run_env(BENCHMARK=1, TAG_ADD="B", COMPLEX=1, INPUT="s9", NOISE=0, ACTION_PENALTY=0.5)

    # C: Random Input with Noise
    run_env(BENCHMARK=1, TAG_ADD="C", COMPLEX=0, INPUT="r", NOISE=0.3, ACTION_PENALTY=0)
    run_env(BENCHMARK=1, TAG_ADD="C", COMPLEX=1, INPUT="r", NOISE=0.3, ACTION_PENALTY=0)

    # D: Seasonal Input with Noise
    run_env(BENCHMARK=1, TAG_ADD="D", COMPLEX=0, INPUT="s9", NOISE=0.2, ACTION_PENALTY=0.5)
    run_env(BENCHMARK=1, TAG_ADD="D", COMPLEX=1, INPUT="s9", NOISE=0.2, ACTION_PENALTY=0.5)

    # Some Experiments
    # run_env(BENCHMARK=1, TAG_ADD="B", COMPLEX=0, INPUT="r", NOISE=0, ACTION_PENALTY=0, MODELS=["PPO"])
    # run_env(BENCHMARK=1, TAG_ADD="D", COMPLEX=1, INPUT="s9", NOISE=0.3, ACTION_PENALTY=0.5, MODELS=["RBA", "A2C"])
    # run_env(LOAD=1, COMPLEX=1, MODELS=["RBA"])

# -------------------------Notes-----------------------------------------------*\
"""
Select a Mode, Model and Logging behaviour by modifying the parameters above.
"""
# -----------------------------------------------------------------------------*\
