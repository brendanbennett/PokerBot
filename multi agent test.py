from collections import deque
from random import randint, random, sample
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from ponk import Ponk

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
        x = Dense(196, activation="relu")(input)
        x = Dense(128, activation='relu')(x)
        output = Dense(2+11, activation="linear")(x)
        #output = concatenate([output_move], axis=1)
        # Call: 0, Fold: 1, Raise: 2-10

        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=Adam(lr=0.001), loss="mse")
        return model

    def remember(self, *args):
        self.memory.append(tuple(args))

    def action_decode(self, q_values):
        m = np.argmax(q_values)
        if m == 0:
            return 0, 0 # call
        elif m == 1:
            return 2, 0 # fold
        else:
            return 1, (m-2)/10# raise


    def next_action(self, state):
        if np.random.rand() < self.exploration_rate:
            q_values = np.random.rand(13,1)
            return self.action_decode(q_values)
        else:
            q_values = self.model.predict(np.array([state,]))
            return self.action_decode(q_values)

    def process(self, states_short_term, winner, folded, winnings):
        # Short term states are (turn, state, action, next_state, money_change, end)
        # To save states as (state, action, next_state, reward)
        # The reward is calculated after the hand is complete
        for s in states_short_term:
            win = True if s[0] == winner else False
            fold = True if s[0] in folded else False
            reward = winnings[s[0]]

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
    agent0 = Agent()
    agent1 = Agent()
    game = Ponk(1)
    game._add_player(1000, "Alice")
    game._add_player(1000, "Bob")
    game._add_player(1000, "Clarisse")

    steps = 0
    show = True
    while True:
        game.verbose = 2 if show else 0
        show = False
        game.start_round()
        states_short_term = []
        w = -1
        state = game.observe()[0]
        while w == -1:
            if steps % 100 == 0:
                show = True
            steps += 1
            if steps % 10 == 0:
                print(steps)
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
        winnings = [p.get_money_diff() for p in game.players]
        agent.process(states_short_term, winner=w, folded=game.get_folded_players(), winnings=winnings)

        game.reset_for_next_hand()

if __name__ == "__main__":
    main()