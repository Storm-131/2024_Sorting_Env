#---------------------------------------------------------*\
# Title: Benchmark (Snippet)
# Author: 
#---------------------------------------------------------*/

def evaluate_model(self, model):
    """Evaluate a model over n episodes and give back the mean and std of the rewards."""
    rewards = []

    for episode in range(self.n_eval_episodes):
        obs, _ = self.eval_env.reset(seed=42+episode)

        # # Optional: Check Sequence of Random Inputs
        # total_values = []
        # for _ in range(10):
        #     _, _, total, _ = self.eval_env.input_generator.generate_input()
        #     total_values.append(total)
        # print(f"Model - Seed {42+episode}:", total_values)

        total_reward = 0
        done = False

        while not done:
            if model is None:
                action = self.train_env.action_space.sample()
            else:
                action = model.predict(obs, deterministic=True)[0]
            obs, reward, done, _, _ = self.eval_env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)

def evaluate_rule_based_agent(self):
    """Evaluate the Rule-Based Agent over n episodes and give back the mean and std of the rewards."""
    rewards = []
    agent = self.agent_model(self.train_env)
    agent.run_analysis()

    for episode in range(self.n_eval_episodes):
        obs, _ = self.eval_env.reset(seed=42+episode)

        # # Optional: Check Sequence of Random Inputs
        # total_values = []
        # for _ in range(10):
        #     _, _, total, _ = self.eval_env.input_generator.generate_input()
        #     total_values.append(total)
        # print(f"RBA - Seed {42+episode}:", total_values)

        total_reward = 0
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.eval_env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\