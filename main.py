import gym

from stable_baselines3 import DQN
from gdi.policies import BoltzmannMlpPolicy

env = gym.make("CartPole-v0")

model = DQN(BoltzmannMlpPolicy, env, policy_kwargs={"tau": 1.0}, verbose=1)
model.learn(total_timesteps=500000, log_interval=4)
model.save("dqn_cartpole")
del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs = env.reset()

total_reward = 0

for episode in range(1, 101):
    episode_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        episode_reward += reward

        if done:
            break

    total_reward += episode_reward
    avg_reward = total_reward / episode
    print(f'episode {episode}: avg_reward={avg_reward}')

    obs = env.reset()
