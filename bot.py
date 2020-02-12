from collections import deque
from random import randint, random, sample
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from ponk2 import Ponk

BATCH_SIZE = 20
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.998

class Agent:
    def __init__(self):
        self.memory = deque(maxlen=1000000)
        self.exploration_rate = 1
        self.model = self.create_model()

    def create_model(self):
        input = Input(shape=(113,))
        x = Dense(128, activation="relu")(input)
        output_move = Dense(3, activation="linear")(x)
        output_raise = Dense(1, activation="relu")(x)
        output = concatenate([output_move, output_raise], axis=1)

        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=Adam(lr=0.001), loss="mse")
        return model

    def remember(self, *args):
        self.memory.append(tuple(args))

    def next_action(self, state):
        if np.random.rand() < self.exploration_rate:
            q_values = np.random.rand(4,1)
            return np.argmax(q_values[:3]), q_values[3][0]
        else:
            q_values = self.model.predict(np.array([state,]))
            return np.argmax(q_values[0][:3]), q_values[0][3]

    def process(self, states_short_term, winner, folded):
        # To save states as (state, action, next_state, reward)
        # The reward is calculated after the hand is complete
        for s in states_short_term:
            win = True if s[0] == winner else False
            fold = True if s[0] in folded else False
            if fold:
                reward = s[4]/4 # Adjust
            else:
                reward = s[4]

            self.remember(s[1], s[2], s[3], reward, s[5])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = sample(self.memory, BATCH_SIZE)
        for state, action, next_state, reward, end in batch:
            next_state = np.array([next_state,])
            state = np.array([state,])
            q_update = reward
            if not end:
                #print(next_state)
                q_update = (reward + GAMMA * np.amax(self.model.predict(next_state)[0][:3]))  # Find optimal Q value.
                # Index is only because predict returns a multidim array but only one input was given, so only one output is given just with shape (1,2)
            q_values = self.model.predict(state)
            q_values[0][action[0]] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def main():
    agent = Agent()
    game = Ponk(1)
    game.add_player(1000, "Alice")
    game.add_player(1000, "Bob")
    game.add_player(1000, "Clarisse")

    while True:
        game.start_round()
        states_short_term = []
        w = -1
        state = game.observe()[0]
        while w == -1:
            turn = game.turn
            action = agent.next_action(state)
            #print(action)
            game.step(action)
            next_state, money_change, w = game.observe()
            #print(w)
            end = False if w is -1 else True
            states_short_term.append((turn, state, action, next_state, money_change, end))
            state = next_state
            #print(state)
            agent.experience_replay()
        agent.process(states_short_term, winner=w, folded=game.get_folded_players())

        game.reset_for_next_hand()

if __name__ == "__main__":
    main()