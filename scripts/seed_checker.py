# ---------------------------------------------------------*\
# Title: Seed Checker
# Author: TM 2024
# ---------------------------------------------------------*/
from src.env_1_simple import SortingEnvironment
from src.env_2_adv import SortingEnvironmentAdv


def test_seed_consistency():
    seed = 42

    # Initialize first environment
    simple_env = SortingEnvironment(max_steps=10, seed=seed, input="r")
    simple_env.reset(seed=seed)

    print("Simple Environment Inputs:")
    for _ in range(2):
        simple_input = simple_env.input_generator.generate_input()
        print(f"Simple Env Input: {simple_input}")

    # Initialize second environment
    adv_env = SortingEnvironmentAdv(max_steps=10, seed=seed, input="r")
    adv_env.reset(seed=seed)

    print("\nAdvanced Environment Inputs:")
    for _ in range(2):
        adv_input = adv_env.input_generator.generate_input()
        print(f"Advanced Env Input: {adv_input}")


# Run the test
test_seed_consistency()


# -------------------------Notes-----------------------------------------------*\
# -> Copy in same hierarchy level as main.py
# This code checks whether both environments (basic and adv) get the same sequence of data
# -----------------------------------------------------------------------------*\
