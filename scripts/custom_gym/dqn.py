#!/usr/bin/env python

import gym 
import envs
import numpy as np
import random
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=200)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0995
        self.learning_rate = 0.01

        self.tau = 0.125

        self.model = self.create_model()
        self.target_model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(12, input_dim=1, activation="relu"))
        model.add(Dense(12))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer = Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 10
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = np.amax(self.target_model.predict(new_state))
                #print(Q_future)
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main():
    env = gym.make("GazeboEnv-v0")
    gamma = 0.9
    epsilon = .95
    trials = 100
    trial_len = 300

    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        this_state = env.reset()
        print("Trial : {}".format(trial))
        for step in range(trial_len):
            print("Trial: {}, Step: {}.".format(trial, step))
            action = dqn_agent.act(this_state)
            new_state, reward, done = env.step(action)

            # reward = reward if not done else -20
            dqn_agent.remember(this_state, action, reward, new_state, done)
            dqn_agent.replay()
            dqn_agent.target_train()
            this_state = new_state
            q_val = dqn_agent.model.predict(this_state)
            print("Q-value for state {} is LEFT:{}, DOWN:{}, RIGHT:{}, UP:{}".format(this_state, q_val[0][0], q_val[0][1], q_val[0][2], q_val[0][3]))
            print("---------------------")  
            if done:
                break

        if step >= 25:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break

if __name__ == "__main__":
    main()

    