from random import choice
from comparator import compare_raw
import numpy as np
import math

VALUES = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
SUITS = ['C','D','H','S']

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
        names = {'C':'Clubs','D':'Diamonds','H':'Hearts','S':'Spades'}
        return self.value + ' ' + names[self.suit]

    def convert(self):
        card = []
        if self.value == '10':
            card.append('T')
        else:
            card.append(self.value)
        card.append(self.suit.lower())
        return ''.join(card)

    def card_to_one_hot(self):
        hot = np.zeros((4,13))
        hot[SUITS.index(self.suit),VALUES.index(self.value)] = 1
        return hot


class Deck:
    def __init__(self):
        self.cards = []
        for value in VALUES:
            for suit in SUITS:
                self.cards.append(Card(value,suit))

    def __str__(self):
        return ', '.join([str(x) for x in self.cards])

    def check_card(self,card):
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
            return np.zeros([4,13])
        return sum([c.card_to_one_hot() for c in self.cards])


class Hand(Deck):
    def __init__(self):
        self.cards = []

class Player:
    def __init__(self, name, money=None):
        self.hand = Hand()
        self.name = name
        self.money = 0
        if money is not None:
            self.money = int(money)
        self.INIT_MONEY = self.money
        self.bet_amount = 0
        self.fold = False


    def deal(self, card):
        self.hand.add_card(card)

    def raise_bet(self, amount):
        if amount <= self.money and amount >= 0:
            self.money -= amount
            self.bet_amount += amount
        else:
            self.bet_amount += self.money
            self.money = 0

    def get_name(self):
        return self.name

    def get_bet_amount(self):
        return self.bet_amount

    def folded(self):
        return self.fold

    def call_to(self, amount):
        self.raise_bet(amount-self.bet_amount)

    def bet_to_pot(self):
        self.bet_amount = 0

    def change_money(self, amount):
        self.money += amount

    def reset_init_money_to_money(self):
        self.INIT_MONEY = self.money

    def get_money(self):
        return self.money

    def get_money_diff(self):
        return self.INIT_MONEY - self.money


class Ponk:
    def __init__(self, smb):
        self.players = []
        self.players_playing = []
        self.start_turn = 0
        self.dealer = 0
        self.turn = 0
        self.pot = 0
        self.deck = Deck()
        self.SMB = smb
        self.com_cards = Hand()
        self.round = 0
        self.num_players = 0
        self.winner = None
        self.check = True
        self.rotations = 0
        self.verbose = 0

    def add_player(self,money,name):
        self.players.append(Player(name,money=money))
        self.players_playing.append(self.num_players)
        self.num_players += 1

    def get_turn_name(self):
        return self.players[self.turn].get_name()

    def get_turn_id(self):
        return self.turn

    def _deal_hands(self):
        for i in range(len(self.players)):
            self.players[i].deal(self.deck.pop_random())
            self.players[i].deal(self.deck.pop_random())

    def next_not_fold(self, p_index):
        for i in range(p_index+1, p_index + self.num_players):
            pm = self.mod(i)
            if not self.players[pm].folded():
                return pm
        return None

    def step_turn(self):
        self.turn = self.mod(self.turn + 1)

    def fold(self, p_index):
        self.players[p_index].fold = True
        self.players_playing.remove(p_index)

    def deal_com_cards(self, n):
        for _ in range(n):
            self.com_cards.add_card(self.deck.pop_random())

    def mod(self,n):
        return n % self.num_players

    def win(self):
        self.give_player_earnings(self.winner)
        print(str(self.players[self.winner].name) + " has won!")

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
            pm = self.mod(p)
            if not self.players[pm].folded():
                return self.players[pm].get_bet_amount()

    def show_all_cards(self):
        all_cards = [self.com_cards.convert()]
        for p_index in self.players_playing:
            hand = self.players[p_index].hand
            all_cards.append(hand.convert())
        return ' '.join(all_cards)

    def compare_hands(self):
        all_cards = self.show_all_cards()
        winner_info = compare_raw(all_cards)
        return winner_info

    def finish_round(self):
        self.check = True
        tot = 0
        for p in self.players:
            tot += p.bet_amount
            p.bet_to_pot()
        self.pot += tot

        if self.winner is not None:
            self.win()
            self.round = 0
        elif self.round == 3:
            self.winner = self.compare_hands()[0]
            self.win()
            self.round = 0
        elif all(self.players[p].money == 0 for p in self.players_playing):
            self.deal_com_cards(5-len(self.com_cards.cards))
            self.winner = self.compare_hands()[0]
            self.win()
            self.round = 0
        else:
            self.round += 1
            self.start_round()

    def give_player_earnings(self, p_index):
        self.players[p_index].change_money(self.pot)
        self.pot = 0

    def game_setup(self):
        self._deal_hands()
        self.players[self.mod(self.dealer + 1)].raise_bet(self.SMB)
        self.players[self.mod(self.dealer + 2)].raise_bet(self.SMB * 2)
        self.start_turn = self.mod(self.dealer + 3)


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
            if self.verbose == 1:
                print(str(self.current_player().name))
            break

    def take_turn(self, action):
        #crf = input(self.get_turn_name() + ": Call, Raise or Fold? (c/r[amount]/f)")
        #crf = self.observe()
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

    def collect_data(self):
        p = self.players[self.turn]

        # 52 for hand, 52 for community cards, 3 for moneys, 3 for bets, 3 for turn (or num players) = 113
        h = p.hand.to_array().ravel()
        c = self.com_cards.to_array().ravel()
        m = np.array([m.money/m.INIT_MONEY for m in self.players])
        b = np.array([i.bet_amount for i in self.players])
        t = np.zeros((self.num_players,))
        t[self.turn] = 1
        data = np.concatenate((h, c, m, b, t))
        return data

    def observe(self):
        self.check_turn()
        w = -1 if self.winner == None else self.winner
        return self.collect_data(), self.players[self.turn].get_money_diff(), w

    def step(self, action):
        if self.verbose == 2:
            self.print_players()
        if self.winner is None:
            if action[0] == 0:
                if self.verbose == 1:
                    print(self.current_player().name + ' called')
                self.take_turn('c')
            elif action[0] == 1:
                r = str(math.ceil(action[1]*self.players[self.turn].money) + (self.SMB*2))
                if self.verbose == 1:
                    print(self.current_player().name + ' raised '+ r)
                self.take_turn('r' + r)
            elif action[0] == 2:
                if self.verbose == 1:
                    print(self.current_player().name + ' folded')
                self.take_turn('f')


    def reset_for_next_hand(self):
        self.winner = None
        self.round = 0
        self.players_playing = []
        for i,p in enumerate(self.players):
            p.fold = False
            p.hand = Hand()
            p.money = p.INIT_MONEY
            p.reset_init_money_to_money()
            self.players_playing.append(i)
        self.dealer = self.mod(self.dealer + 1)
        self.deck = Deck()
        self.com_cards = Hand()

#'''

#'''


#for i in ponker.current_player_show():
#    print(i)
#game = Ponk(1)
#game.add_player(1000, "Alice")
#game.add_player(1000, "Bob")
#game.add_player(1000, "Clarisse")
#game.start_round()

#while True:
#    print(game.observe())
#    game.step('c')