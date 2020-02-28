from collections import deque
from random import sample, randrange
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from ponk2 import Ponk
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 20
GAMMA_MIN = 0.6
GAMMA_MAX = 0.95
GAMMA_GROWTH = 0.005
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.999

RANDOM_TOURNAMENT_INTERVAL = 30


def action_decode(q_values):
    m = np.argmax(q_values)
    if m == 0:
        return 0, 0  # call
    elif m == 1:
        return 2, 0  # fold
    else:
        return 1, (m - 2) / 10  # raise


class Agent:
    def __init__(self, num_players):
        self.memory = deque(maxlen=1000000)
        self.exploration_rate = EXPLORATION_MAX
        self.gamma = GAMMA_MIN
        self.num_players = num_players
        self.model = self.create_model()

    def create_model(self):
        input_layer = Input(shape=(104 + (3 * self.num_players),))
        x = Dense(192, activation="relu")(input_layer)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(2 + 11, activation="linear")(x)
        # output = concatenate([output_move], axis=1)
        # Call: 0, Fold: 1, Raise: 2-10

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(lr=0.001), loss="mse")
        return model

    def remember(self, tup):
        self.memory.append(tup)

    def next_action(self, state):
        if np.random.rand() < self.exploration_rate:
            q_values = np.random.rand(13, 1)
            return action_decode(q_values)
        else:
            q_values = self.model.predict(np.array([state, ]))
            return action_decode(q_values)

    def process(self, states_short_term, winner, winnings):
        # Short term states are (turn, state, action, next_state, money_change, end)
        # To save states as (state, action, next_state, reward)
        # The reward is calculated after the hand is complete
        winner_final_turn = None
        for i in range(len(states_short_term) - 1, -1, -1):
            if states_short_term[i][0] == winner:
                winner_final_turn = i
                # print('Player {} won on turn {}'.format(states_short_term[i][0], winner_final_turn))
                break

        for i, s in enumerate(states_short_term):
            if i == winner_final_turn:
                self.remember((s[1], s[2], s[3], s[4] + winnings[winner], s[5]))
                # print('Player {} is rewarded {}'.format(s[0],s[4]+winnings[winner]))

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = sample(self.memory, BATCH_SIZE)
        for state, action, next_state, reward, end in batch:
            next_state = np.array([next_state, ])
            state = np.array([state, ])
            q_update = reward
            if not end:
                # print(next_state)
                q_update = (reward + self.gamma * np.amax(self.model.predict(next_state)[0][:3]))  # Find optimal Q
                # Index is only because predict returns a multi-dim array but only one input was given, so only one
                # output is given just with shape (1,2)
            else:
                q_update = reward
            q_values = self.model.predict(state)
            q_values[0][action[0]] = q_update
            self.model.fit(state, q_values, verbose=0)

        self.gamma *= GAMMA_GROWTH
        self.gamma = min(GAMMA_MAX, self.gamma)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


class RandomAgent:
    @staticmethod
    def next_action():
        actions = np.random.rand(13, 1)
        actions[2:] = actions[2:] * 0.1  # Normalise so all (call raise fold) actions are just as likely
        return action_decode(actions)


class ScoreLogger:
    def __init__(self):
        self.scores = []
        self.steps = []

    def log(self, run, step):
        self.scores.append(run)
        self.steps.append(step)

    def show(self):
        sns.scatterplot(x=self.steps, y=self.scores)
        plt.show()


def main():
    log = ScoreLogger()
    agent = Agent(4)
    rand_agent = RandomAgent()
    game = Ponk(1)
    game.add_player(1000, "Alice", bot=True)
    game.add_player(1000, "Bob", bot=True)
    game.add_player(1000, "Clarisse", bot=True)
    game.add_player(1000, "Dave", bot=True)

    steps = 0
    game.verbose = 0
    random_rounds = 0
    num_tours = 1
    while True:
        if (game.hand_num // game.num_players) * game.num_players % RANDOM_TOURNAMENT_INTERVAL == 0:
            random_rounds += 1
            # Choose random agents
            bot_index = randrange(3)
            print("RANDOM TOURNAMENT GAME for " + game.players[bot_index].name)

            game.start_round()
            w = -1
            state = game.observe()[0]
            while w == -1:
                if steps % 100 == 0:
                    print('Steps = ' + str(steps))
                turn = game.turn

                if turn == bot_index:
                    action = agent.next_action(state)
                else:
                    action = rand_agent.next_action()
                game.display()
                state, _, w = game.step(action, show=True)
            winnings = [p.get_money_diff() for p in game.players]
            log.log(winnings[bot_index], steps)
            if random_rounds == game.num_players:
                num_tours += 1
                random_rounds = 0
            if num_tours % 10 == 0:
                log.show()

        else:
            game.start_round()
            states_short_term = []
            w = -1
            state = game.observe()[0]
            while w == -1:
                steps += 1
                if steps % 100 == 0:
                    print('Steps = ' + str(steps))
                turn = game.turn

                if game.current_player().bot:
                    action = agent.next_action(state)
                else:
                    action = input(game.current_player().name + "'s turn: ")
                if False:  # game.hand_num % 100 == 0:
                    game.display()
                    next_state, instant_money_change, w = game.step(action, show=True)
                else:
                    next_state, instant_money_change, w = game.step(action, show=False)
                end = False if w is -1 else True
                if game.current_player().bot:
                    states_short_term.append((turn, state, action, next_state, instant_money_change, end))
                state = next_state
                agent.experience_replay()
            final_changes = [p.instant_change for p in game.players]
            agent.process(states_short_term, w, final_changes)

        game.reset_for_next_hand()


if __name__ == "__main__":
    main()
