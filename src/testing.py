# ---------------------------------------------------------*\
# Title: Testing
# Author: TM 05.2024
# ---------------------------------------------------------*/

# ---------------------------------------------------------*/
# Test the Environment with Random Actions
# ---------------------------------------------------------*/


def test_env(env=None, tag="", save=False, title="", steps=50, dir="./img/", video=False, seed=None):
    """Test the environment by running a random simulation for a given number of steps."""
    if env is None:
        raise ValueError("Environment must be provided")

    if seed is not None:
        env.action_space.seed(seed)
        
    obs, _ = env.reset(seed=seed)
    print("Initial Observation:", obs)

    for i in range(steps):
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, _, _ = env.step(action)
        # print(f"Step {i+1} - Action: {action}, Observation: {obs}, Reward: {reward}")
        if done:
            env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title)
            obs, _ = env.reset()

# ---------------------------------------------------------*/
# Test the Model in the Environment
# ---------------------------------------------------------*/


def test_model(model, env=None, tag="", save=False, title="", steps=50, dir="./img/", seed=42):
    """Test the model in the environment by running a simulation for a given number of steps."""
    if env is None:
        raise ValueError("Environment must be provided")

    obs, _ = env.reset(seed=seed)
    print("Initial Observation:", obs)

    for i in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        # print(f"Step {i+1} - Action: {action}, Next Observation: {obs}, Reward: {reward}")
        if done:
            env.render(save=save, log_dir=dir, filename=f'{tag}_{model.__class__.__name__}_simulation', title=title)
            obs, _ = env.reset()


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
