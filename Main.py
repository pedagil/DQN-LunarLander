import gym
import numpy as np
import matplotlib.pyplot as plt
from Agent import DQN

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    print(env.observation_space)
    print(env.action_space)
    episodes = 400
    renderAll = False

    agent = DQN(env.action_space.n, env.observation_space.shape[0], env)
    rewards, mean_reward, epsilon = agent.train_dqn(episodes, renderAll)

    fig1 = plt.subplots()
    plt.plot([i for i in range(0, len(rewards))], rewards, 'ro', markersize=1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards')

    fig2 = plt.figure()
    plt.plot([i for i in range(0, len(mean_reward))], mean_reward)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Mean reward over last 100 episodes')

    fig3 = plt.figure()
    plt.plot([i for i in range(0, len(epsilon))], epsilon)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon of e-greedy policy')

    plt.show()
    env.close()
