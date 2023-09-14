from collections import deque
from random import sample, randrange
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np
from ponk import Ponk, PonkConfig
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir, mkdir
import json
from time import sleep
import re

TRAINING_DIR = "training/"
DEFAULT_SAVE = "save"
BATCH_SIZE = 20
GAMMA_MIN = 0.6
GAMMA_MAX = 0.95
GAMMA_GROWTH = 1.005
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

def create_save_files(dir,steps,num_tours,hand_num,log,model,name=None):
    saves = listdir(TRAINING_DIR)
    if name is None:
        nums = []
        for s in saves:
            if re.search(rf"^{DEFAULT_SAVE}\d+",s) is not None:
                nums.append(int(s[len(DEFAULT_SAVE):]))
        if len(nums) > 0:
            next = max(nums)+1
        else:
            next = 1
        new_save_dir = TRAINING_DIR + DEFAULT_SAVE + str(next) + "/"
    else:
        new_save_dir = TRAINING_DIR + name + "/"
        if name not in saves:
            mkdir(new_save_dir)


    info = {
        "steps": steps,
        "hand_num": hand_num,
        "num_tours": num_tours
    }
    model.save(new_save_dir + "model.h5")
    with open(new_save_dir + "info.json", "w+") as f:
        json.dump(info, f)

    with open(new_save_dir + "log.json", "w+") as f:
        json.dump({"scores": log.scores, "steps": log.steps}, f)

    print("Saved model")


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

    def save(self, info):
        self.model.save(TRAINING_DIR+"model.h5")
        with open(TRAINING_DIR+"info.json", "w+") as f:
            json.dump(info, f)
        print("Saved model")


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
        sleep(2)
        plt.close('all')

    def save(self):
        with open(TRAINING_DIR+"log.json", "w+") as f:
            json.dump({"scores":self.scores,"steps":self.steps},f)

    def load(self):
        with open(TRAINING_DIR+"log.json", "r") as f:
            d = json.load(f)
            self.scores, self.steps = (d["scores"], d["steps"])


def main():
    log = ScoreLogger()
    agent = Agent(4)
    test_agent = Agent(4)
    config = PonkConfig(num_players=4, small_blind=1, starting_money=100, verbose=0)
    game = Ponk(config)

    random_rounds = 0
    save_next = False
    steps = 0
    num_tours = 1

    # file system
    saved = listdir(TRAINING_DIR)
    save_name = ""

    if len(saved) > 0:
        print("Saves available: "+", ".join(saved))
        while True:
            save_name = input("Type save to work with or press enter to make a new one:")
            if save_name == "":
                break
            elif save_name in saved:
                break
            else:
                print("Please select a valid file.")

    if len(save_name) == 0:
        while True:
            save_name = input("Type name for new save:")
            if save_name in saved:
                print("Please pick a unique name.")
            else:
                break

    try:
        if "model.h5" in listdir(TRAINING_DIR+save_name+"/"):
            with open(TRAINING_DIR+save_name+"/info.json", "r") as f:
                info = json.load(f)
            u = input("Continue training with {} steps? (y/n)".format(info["steps"]))
            if u == "y":
                agent.model = load_model(TRAINING_DIR+save_name+"/model.h5")
                steps = info["steps"]
                num_tours = info["num_tours"]
                game.hand_num = info["hand_num"]
                log.load()
    except FileNotFoundError:
        pass

    while True:
        test_name = input("Select testing model {}:".format(", ".join(saved)))
        if test_name in saved:
            break
        else:
            print("Please select a valid file.")

    test_agent.model = load_model(TRAINING_DIR + test_name + "/model.h5")

    while True:
        if (game.hand_num // game.num_players) * game.num_players % RANDOM_TOURNAMENT_INTERVAL == 0:
            random_rounds += 1
            # Choose testing agents
            bot_index = randrange(3)
            print("TOURNAMENT GAME for " + game.players[bot_index].name)

            state = game.reset()
            w = -1
            while w == -1:
                if steps % 100 == 0:
                    print('Steps = ' + str(steps))
                turn = game.turn

                if turn == bot_index:
                    action = agent.next_action(state)
                else:
                    action = test_agent.next_action(state)
                game.display()
                state, _, w = game.step(action, show=True)
            winnings = [p.get_money_diff() for p in game.players]
            log.log(winnings[bot_index], steps)
            if random_rounds == game.num_players:
                num_tours += 1
                random_rounds = 0
                #if num_tours % 1 == 0:
                #    log.show()

        else:
            state = game.reset()
            states_short_term = []
            w = -1
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
                if steps%1000 == 0:
                    save_next = True
            final_changes = [p.instant_change for p in game.players]
            agent.process(states_short_term, w, final_changes)
        game.reset_for_next_hand()

        if save_next:
            create_save_files(TRAINING_DIR,steps,num_tours,game.hand_num,log,agent.model,name=save_name)
            save_next = False


if __name__ == "__main__":
    main()
