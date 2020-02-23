import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
from keras.activations import relu, linear
import numpy as np

USE_TARGET_NETWORK = False
UPDATE_TARGET_NETWORK = 0
STOP_AFTER_N_EPISODES = 1000
RENDER_EVERY = 10


class DQN:

    def __init__(self, action_space, state_space, env):

        self.env = env
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.learning_rate = 0.001
        self.epsilon_decay = .99
        self.replay_buffer = deque(maxlen=1000000)

        self.model = self.build_model()

        if USE_TARGET_NETWORK:
            self.target_update_counter = 0
            self.target_model = self.build_model()
            self.target_model.set_weights(self.model.get_weights())

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(128, activation=relu))
        # model.add(Dense(128, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
        return model

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values)

    def replay(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([i[0] for i in mini_batch])
        actions = np.array([i[1] for i in mini_batch])
        rewards = np.array([i[2] for i in mini_batch])
        next_states = np.array([i[3] for i in mini_batch])
        dones = np.array([i[4] for i in mini_batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        if USE_TARGET_NETWORK:
            targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (
                    1 - dones)
        else:
            targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (
                    1 - dones)

        targets_full = self.model.predict_on_batch(states)

        indexes = np.array([i for i in range(self.batch_size)])
        targets_full[[indexes], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0, shuffle=False)

    def train_dqn(self, renderAll):
        render = False
        loss = []
        mean_reward_list = []
        epsilon_list = []
        episode_counter = 1

        while True:
            # for e in range(episode):
            state = self.env.reset()
            state = np.reshape(state, (1, self.state_space))
            score = 0
            max_steps = 3000
            for i in range(max_steps):
                action = self.act(state)
                if episode_counter % RENDER_EVERY == 1 or renderAll:
                    self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                next_state = np.reshape(next_state, (1, self.state_space))
                self.add_to_replay_buffer(state, action, reward, next_state, done)
                state = next_state
                self.replay()
                if done:
                    if episode_counter % RENDER_EVERY == 1:
                        print("In episode", episode_counter, "reward is", score)
                    break
            loss.append(score)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                # print("Epsilon is {}".format(self.epsilon))

            if USE_TARGET_NETWORK:
                if done:
                    self.target_update_counter += 1

                if self.target_update_counter > UPDATE_TARGET_NETWORK:
                    self.target_model.set_weights(self.model.get_weights())
                    self.target_update_counter = 0

            # Average score of last 100 episode
            is_solved = np.mean(loss[-100:])

            if episode_counter % RENDER_EVERY == 1:
                if episode_counter < 100:
                    print("Average over last", episode_counter, "episodes:", is_solved, "\n")
                else:
                    print("Average over last 100 episodes:", is_solved, "\n")
                if is_solved > 200 and len(loss[-100:]) == 100:
                    print('\n Task Completed! \n')
                    break

            mean_reward_list.append(is_solved)
            epsilon_list.append(self.epsilon)

            if episode_counter == STOP_AFTER_N_EPISODES:
                break
            episode_counter += 1
        return loss, mean_reward_list, epsilon_list
