import random
from collections import deque

import matplotlib.pyplot as plt
import seaborn as sns
import gym
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, Add
from keras.models import Model
from keras.optimizers import Adam

# from scores.score_logger import ScoreLogger

ENV_NAME = "Pendulum-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.998


class ScoreLogger:
    def __init__(self):
        self.runs = []
        self.steps = []

    def log(self, run, step):
        self.runs.append(run)
        self.steps.append(step)


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        _,self.actor = self.create_actor()
        _,_,self.critic = self.create_critic()

        actor_weights = self.actor.trainable_weights
        # Find gradients with respect to the weights
        self.actor_grads = tf.gradients(self.actor_model.output,actor_weights)


    def create_actor(self):
        state_input = Input(shape=self.observation_space.shape)
        h = Dense(32,activation='relu')(state_input)
        output = Dense(self.action_space.shape[0],activation='tanh')
        model = Model(input=state_input,output=output)
        model.compile(loss='mse', optimiser=Adam(lr=LEARNING_RATE))
        return state_input, model

    def create_critic(self):
        state_input = Input(shape=self.observation_space.shape)
        action_input = Input(shape=self.action_space.shape)
        hs = Dense(16,activation='relu')(state_input)
        ha = Dense(16,activation='relu')(action_input)

        h1 = Add()([hs,ha])
        h2 = Dense(32,activation='relu')(h1)
        output = Dense(self.action_space.shape[0],activation='tanh')(h2)
        model = Model(input=[state_input,action_input],output=output)
        model.compile(loss='mse', optimiser=Adam(lr=LEARNING_RATE))
        return state_input, action_input, model

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, state_next, terminal in batch:
            if not terminal:
                predicted_action = self.actor.predict(state_next)
                future_reward = self.critic.predict([state_next,predicted_action])
                reward += GAMMA*future_reward
            self.critic.fit([state,action], reward)

        for state, action, reward, state_next, terminal in batch:
            if not terminal:



def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger()
    observation_space = env.observation_space
    action_space = env.action_space
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print(
                    "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.log(run, step)
                break
            dqn_solver.experience_replay()
        if run % 20 == 0:
            sns.scatterplot(x=score_logger.runs, y=score_logger.steps)
            plt.show()


if __name__ == "__main__":
    cartpole()
