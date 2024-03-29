from random import choice
from comparator import compare_raw
import numpy as np
import math

VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['C', 'D', 'H', 'S']

class PonkConfig():
    def __init__(self, num_players: int, small_blind: int, starting_money: int, verbose: int = 0) -> None:
        self.num_players: int = num_players
        self.small_blind: int = small_blind
        self.starting_money: int = starting_money
        self.verbose = verbose

class Card:
    def __init__(self, value, suit):
        if value in VALUES:
            self.value = value
        else:
            raise Exception("Invalid Value")
        if suit in SUITS:
            self.suit = suit
        else:
            raise Exception("Invalid Suit")

    def __eq__(self, other):
        if self.suit == other.suit and self.value == other.value:
            return True
        return False

    def __str__(self):
        names = {'C': 'Clubs', 'D': 'Diamonds', 'H': 'Hearts', 'S': 'Spades'}
        return self.value + ' ' + names[self.suit]

    def card_to_one_hot(self):
        hot = np.zeros((4, 13))
        hot[SUITS.index(self.suit), VALUES.index(self.value)] = 1
        return hot

    def convert(self):
        card = []
        if self.value == '10':
            card.append('T')
        else:
            card.append(self.value)
        card.append(self.suit.lower())
        return ''.join(card)

    def display_card(self):
        # ┌──┐
        # │3♠│
        # └──┘
        c = ['┌──┐',
             '│' + self.convert()[0] + ['♣', '♦', '♥', '♠'][SUITS.index(self.convert()[1].upper())] + '│',
             '└──┘']
        return c


class Hand:
    def __init__(self):
        self.cards = []

    def __str__(self):
        return ', '.join([str(x) for x in self.cards])

    def check_card(self, card):
        if card in self.cards:
            return True
        return False

    def pop(self, card):
        if self.check_card(card):
            self.cards.remove(card)
            return card
        else:
            return False

    def pop_random(self):
        return self.pop(choice(self.cards))

    def add_card(self, card):
        if card not in self.cards:
            self.cards.append(card)
        else:
            raise Exception("Card already in deck!")

    def convert(self):
        return ''.join([c.convert() for c in self.cards])

    def to_array(self):
        if len(self.cards) == 0:
            return np.zeros([4, 13])
        return sum([c.card_to_one_hot() for c in self.cards])


class Deck(Hand):
    def __init__(self):
        super().__init__()
        for value in VALUES:
            for suit in SUITS:
                self.cards.append(Card(value, suit))


class Player:
    def __init__(self, name, money=None):
        self.hand = Hand()
        self.name = name
        self.money = 0
        if money is not None:
            self.money = int(money)
        self._init_money = self.money
        self.bet_amount = 0
        self.is_folded = False
        self.instant_change = 0

    def reset_for_next_hand(self):
        self.instant_change = 0
        self.bet_amount = 0
        self.hand = Hand()
        self.is_folded = False

    def get_name(self):
        return self.name

    def get_bet_amount(self):
        return self.bet_amount

    def deal(self, card):
        self.hand.add_card(card)

    def change_money(self, amount):
        self.money += amount
        self.instant_change = amount

    def raise_bet(self, amount):
        if self.money >= amount >= 0:
            self.change_money(-amount)
            self.bet_amount += amount
        else:
            self.bet_amount += self.money
            self.change_money(-self.money)

    def call_to(self, amount):
        self.raise_bet(amount - self.bet_amount)

    def bet_to_zero(self):
        self.bet_amount = 0

    def reset_init_money_to_money(self):
        self._init_money = self.money

    def get_money(self):
        return self.money

    def get_money_diff(self):
        return self._init_money - self.money

    def fold(self):
        self.is_folded = True
        self.instant_change = 0

    def folded(self):
        return self.is_folded


class Ponk:
    def __init__(self, config: PonkConfig):
        self._config = config

    def reset(self):
        self.players: list[Player] = list()
        self.players_playing = list()
        self.start_turn = 0
        self.dealer = 0
        self.turn = 0
        self.pot = 0
        self.deck = Deck()
        self.small_blind = self._config.small_blind
        self.com_cards = Hand()
        self.round = 0
        self.num_players = self._config.num_players
        self.winner = None
        self.check = True
        self.rotations = 0
        self.verbose = self._config.verbose
        self.total_turns_in_hand = 0
        self.hand_num = 0
        for i in range(self._config.num_players):
            self._add_player(self._config.starting_money, name=str(i))
        self._reset_players_playing()

        # Initialise round
        self.start_round()

        # Return initial state
        return self.observe()[0]

    def _add_player(self, money, name):
        """Only to be called on game creation"""
        self.players.append(Player(name, money=money))

    def _reset_players_playing(self):
        self.players_playing = list(range(len(self.players)))

    def _mod(self, n):
        return n % self.num_players

    def get_turn_name(self):
        return self.players[self.turn].get_name()

    def step_turn(self):
        self.turn = self._mod(self.turn + 1)
        self.total_turns_in_hand += 1

    def _deal_hands(self):
        for i in range(len(self.players)):
            self.players[i].deal(self.deck.pop_random())
            self.players[i].deal(self.deck.pop_random())

    def deal_com_cards(self, n):
        for _ in range(n):
            self.com_cards.add_card(self.deck.pop_random())

    def fold(self, p_index):
        self.players[p_index].fold()
        if self.verbose == 2:
            print(f"players_playing = {self.players_playing}. Removing {p_index}.")
        self.players_playing.remove(p_index)

    def next_not_fold(self, p_index):
        for i in range(p_index + 1, p_index + self.num_players):
            pm = self._mod(i)
            if not self.players[pm].folded():
                return pm
        return None

    def win(self):
        self.give_player_earnings(self.winner)
        if self.verbose >= 1:
            print(str(self.players[self.winner].name) + " has won hand " + str(self.hand_num))

    def current_player(self):
        return self.players[self.turn]

    def show_player(self, p_index):
        fold = '-Folded' if self.players[p_index].folded() else ''
        return [self.players[p_index].name + fold + " " +
                str(self.players[p_index].money) + " " +
                str(self.players[p_index].bet_amount),
                self.players[p_index].hand, self.com_cards]

    def current_player_show(self):
        return [self.get_turn_name() + " " +
                str(self.players[self.turn].money) + " " +
                str(self.players[self.turn].bet_amount),
                self.players[self.turn].hand, self.com_cards]

    def print_players(self):
        for i in range(self.num_players):
            for j in self.show_player(i):
                print(j)
        print()

    def get_folded_players(self):
        folded = []
        for i in range(self.num_players):
            if i not in self.players_playing:
                folded.append(i)
        return folded

    def check_equal_bets(self):
        bet = self.players[self.players_playing[0]].bet_amount
        allin = False
        for i in self.players_playing:
            if self.players[i].bet_amount != bet:
                if self.players[i].money == 0:
                    allin = True
                    break
                return False
        if allin:
            for i in self.players_playing:
                if self.players[i].money != 0:
                    return False
            return True
        return True

    def get_previous_bet(self, p_index):
        for p in range(p_index - 1, p_index - self.num_players, -1):
            pm = self._mod(p)
            if not self.players[pm].folded():
                return self.players[pm].get_bet_amount()

    def convert_all_cards(self):
        all_cards = [self.com_cards.convert()]
        for p_index in self.players_playing:
            hand = self.players[p_index].hand
            all_cards.append(hand.convert())
        return ' '.join(all_cards)

    def compare_hands(self):
        all_cards = self.convert_all_cards()
        # print('Showdown: ', all_cards)
        # print([self.players[p].name for p in self.players_playing])
        winner_info = compare_raw(all_cards)
        return winner_info

    def give_player_earnings(self, p_index):
        self.players[p_index].change_money(self.pot)
        self.pot = 0

    def game_setup(self):
        self._deal_hands()
        self.players[self._mod(self.dealer + 1)].raise_bet(self.small_blind)
        self.players[self._mod(self.dealer + 2)].raise_bet(self.small_blind * 2)
        self.start_turn = self._mod(self.dealer + 3)
        self.total_turns_in_hand = 0
        self.hand_num += 1

    def check_turn(self):
        while self.winner is None:
            if self.turn == self.start_turn:
                self.rotations += 1
                if self.rotations == 2 and self.check:
                    self.finish_round()
                    continue
            if len(self.players_playing) == 1:
                self.winner = self.players_playing[0]
                self.finish_round()
                continue
            if self.players[self.turn].folded():
                self.step_turn()
                continue
            if self.check_equal_bets() and self.players[self.next_not_fold(self.dealer)].bet_amount != 0 or (
                    self.check_equal_bets() and not self.check):
                self.finish_round()
                continue
            if self.players[self.turn].money == 0:
                self.step_turn()
                continue
            break

    def take_turn(self, action):
        if action == 'f':
            self.fold(self.turn)
        elif action == 'c':
            bet = self.get_previous_bet(self.turn)
            self.players[self.turn].raise_bet(bet - self.players[self.turn].bet_amount)
        elif action[0] == 'r':
            if self.check:
                self.check = False
            r = int(action[1:])
            self.players[self.turn].raise_bet(r + self.get_previous_bet(self.turn) - self.players[self.turn].bet_amount)
        else:
            raise Exception
        self.step_turn()

    def start_round(self):
        self.winner = None
        self.check = True
        if self.round == 0:
            self.game_setup()
            self.check = False
        elif self.round == 1:
            self.start_turn = self.next_not_fold(self.dealer)
            self.deal_com_cards(3)
        elif self.round == 2 or self.round == 3:
            self.start_turn = self.next_not_fold(self.dealer)
            self.deal_com_cards(1)
        self.turn = self.start_turn
        self.rotations = 0

    def finish_round(self):
        self.check = True
        tot = 0
        for p in self.players:
            tot += p.bet_amount
            p.bet_to_zero()
        self.pot += tot

        if self.winner is not None:
            self.win()
            self.round = 0
        elif self.round == 3:
            self.winner = self.players_playing[self.compare_hands()[0]]
            self.win()
            self.round = 0
        elif all(self.players[p].money == 0 for p in self.players_playing):
            self.deal_com_cards(5 - len(self.com_cards.cards))
            self.winner = self.players_playing[self.compare_hands()[0]]
            self.win()
            self.round = 0
        else:
            self.round += 1
            self.start_round()

    def collect_data(self):
        p: Player = self.players[self.turn]

        # 52 for hand, 52 for community cards, n for moneys, n for bets, n for turn (for n num players = 104 + 3n)
        h = p.hand.to_array().ravel()
        c = self.com_cards.to_array().ravel()
        m = np.array([p.money / p._init_money for p in self.players])
        b = np.array([p.bet_amount / p._init_money for p in self.players])
        t = np.zeros((self.num_players,))
        t[self.turn] = 1
        data = np.concatenate((h, c, m, b, t))
        return data

    def observe(self):
        reward = self.players[self._mod(self.turn - 1)].instant_change
        # print('Player'+str(self.mod(self.turn-1)) + ' reward '+str(reward))
        self.check_turn()
        w = -1 if self.winner is None else self.winner
        return self.collect_data(), reward, w

    def step(self, action, show=False):
        if self.winner is not None:
            raise Exception("Tried to step after game won")
        
        if action[0] == 0:
            if show:
                print(self.current_player().name + ' called')
            self.take_turn('c')
        elif action[0] == 1:
            # TODO make raises in terms of small blinds
            r = str(math.ceil(action[1] * self.players[self.turn].money) + (self.small_blind * 2))
            if show:
                print(self.current_player().name + ' raised ' + r) 
            self.take_turn('r' + r)
        elif action[0] == 2:
            if show:
                print(self.current_player().name + ' folded')
            self.take_turn('f')

        return self.observe()

    def reset_for_next_hand(self):
        self.winner = None
        self.round = 0
        for i, p in enumerate(self.players):
            p.is_folded = False
            p.reset_for_next_hand()
            # I don't why I did this
            # p.money = p._init_money
            # p.reset_init_money_to_money()
        self._reset_players_playing()
        self.dealer = self._mod(self.dealer + 1)
        self.deck = Deck()
        self.com_cards = Hand()

        # Initialise round
        self.start_round()

        # Return initial state
        return self.observe()[0]

    def display(self):
        print('═' * 30)
        print('Hand ' + str(self.hand_num) + '   Round ' + str(self.round) + '   Turn in hand: ' + str(
            self.total_turns_in_hand))
        names = [p.name for p in self.players]
        for i, name in enumerate(names):
            t = '•' if self.turn == i else ' '
            if i in self.players_playing:
                names[i] = t + names[i] + ''.join([' '] * (11 - len(name)))
            else:
                names[i] = t + names[i] + ''.join(['X'] * (11 - len(name)))
        print(' '.join(names))
        cards = [p.hand.cards for p in self.players]
        cards_flat = []
        for i in cards:
            for c in i:
                cards_flat.append(c)
        cards = [c.display_card() for c in cards_flat]
        for r in range(3):
            for i, c in enumerate(cards):
                if i % 2 == 0:
                    print(c[r], end='  ')
                else:
                    print(c[r], end='   ')
            print()

        moneys = [str(p.money) + '  ' + str(p.bet_amount) for p in self.players]
        for i, money in enumerate(moneys):
            moneys[i] += ''.join([' '] * (12 - len(money)))
        print('', ' '.join(moneys))

        cards = [c.display_card() for c in self.com_cards.cards]
        for r in range(3):
            for c in cards:
                print(c[r], end='  ')
            print()

        print('-' * 30)

def human_friendly_input(game: Ponk):
    _decoder = {"c": 0, "r": 1, "f": 2} # TODO make this friendlier
    
    print(game.current_player().name + "'s turn")
    while True:
        action_type = input("Choose action ([c]: call, [r] raise, [f]: fold):")
        if action_type not in _decoder.keys():
            print("Not a valid action!")
            continue
        break

    raise_amount = 0
    if action_type == "r":
        while True:
            raise_amount = input(f"fraction to raise (have {game.current_player().money}):")
            try:
                raise_amount = float(raise_amount)
            except TypeError:
                print("Not a valid number!")
                continue
            break
    return _decoder[action_type], raise_amount
    

if __name__ == "__main__":
    config = PonkConfig(num_players=4, small_blind=1, starting_money=100, verbose=2)
    game = Ponk(config)
    state = game.reset()
    while True:
        w = -1
        while w == -1:
            game.display()
            action = human_friendly_input(game)
            _, _, w = game.step(action, show = True)
        game.reset_for_next_hand()
