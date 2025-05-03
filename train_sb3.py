"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    train_env = gym.make('CustomHopper-source-v0')
    timesteps = 1000000
    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #
    model = PPO("MlpPolicy", train_env, verbose = 1, gamma=0.99, n_steps=4096, batch_size=128, learning_rate=0.0003)
    model.learn(total_timesteps=timesteps)
    model.save("ppo_hopper")
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == '__main__':
    main()
