from random import choice
from pokerbot.comparator import compare_raw
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
            raise ValueError(f"Invalid value: {value!r}")
        if suit in SUITS:
            self.suit = suit
        else:
            raise ValueError(f"Invalid suit: {suit!r}")

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
            raise ValueError("Card already in deck!")

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
        self.hand_contribution = 0
        self.is_folded = False
        self.instant_change = 0

    def reset_for_next_hand(self):
        self.instant_change = 0
        self.bet_amount = 0
        self.hand_contribution = 0
        self.hand = Hand()
        self.is_folded = False
        self._init_money = self.money

    def get_name(self):
        return self.name

    def get_bet_amount(self):
        return self.bet_amount

    def deal(self, card):
        self.hand.add_card(card)

    def change_money(self, amount):
        self.money += amount
        self.instant_change += amount

    def raise_bet(self, amount):
        amount = max(amount, 0)
        actual = min(amount, self.money)
        self.bet_amount += actual
        self.hand_contribution += actual
        self.change_money(-actual)

    def call_to(self, amount):
        self.raise_bet(amount - self.bet_amount)

    def bet_to_zero(self):
        self.bet_amount = 0

    def reset_init_money_to_money(self):
        self._init_money = self.money

    def get_money(self):
        return self.money

    def get_money_diff(self):
        return self.money - self._init_money

    def fold(self):
        self.is_folded = True
        self.instant_change = 0

    def folded(self):
        return self.is_folded


class Ponk:
    def __init__(self, config: PonkConfig):
        self._config = config
        self.num_players = self._config.num_players
        self.small_blind = self._config.small_blind
        self.verbose = self._config.verbose
        self.hand_num = 0
        self.dealer = 0
        self.players: list[Player] = []
        for i in range(self._config.num_players):
            self._add_player(self._config.starting_money, name=str(i))

    def reset(self):
        """Reset per-hand state. Preserves players, money, hand_num, and dealer."""
        for p in self.players:
            p.reset_for_next_hand()
        self._reset_players_playing()
        self.start_turn = 0
        self.turn = 0
        self.pot = 0
        self.deck = Deck()
        self.com_cards = Hand()
        self.round = 0
        self.winner = None
        self.check = True
        self.rotations = 0
        self.total_turns_in_hand = 0
        self.start_round()
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

    def _distribute_pot(self):
        """Award self.pot among non-folded players, building side pots from
        each distinct contribution level so all-in players can only win up to
        what they contributed."""
        contributions = [p.hand_contribution for p in self.players]
        eligible = [i for i in range(self.num_players) if not self.players[i].folded()]
        if not eligible:
            self.pot = 0
            return

        if len(eligible) == 1:
            winner = eligible[0]
            self.players[winner].change_money(self.pot)
            self.winner = winner
            self.pot = 0
            return

        levels = sorted({contributions[i] for i in eligible})
        prev_level = 0
        main_winner = None
        for level in levels:
            side_pot = sum(min(c, level) - min(c, prev_level) for c in contributions)
            prev_level = level
            if side_pot == 0:
                continue
            contenders = [i for i in eligible if contributions[i] >= level]
            if len(contenders) == 1:
                winners = contenders
            else:
                all_cards = [self.com_cards.convert()]
                for i in contenders:
                    all_cards.append(self.players[i].hand.convert())
                best, choppers = compare_raw(' '.join(all_cards))
                winners = [contenders[idx] for idx in [best, *choppers]]

            share, remainder = divmod(side_pot, len(winners))
            for idx, w in enumerate(winners):
                bonus = remainder if idx == 0 else 0
                self.players[w].change_money(share + bonus)
            if main_winner is None:
                main_winner = winners[0]

        self.winner = main_winner if main_winner is not None else eligible[0]
        self.pot = 0
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
        # All players who still have chips to commit must match the highest bet.
        # All-in players are exempt from matching, but their bet still counts
        # toward the target (an all-in raise must be responded to).
        if not self.players_playing:
            return True
        target = max(self.players[i].bet_amount for i in self.players_playing)
        for i in self.players_playing:
            p = self.players[i]
            if p.money == 0:
                continue
            if p.bet_amount != target:
                return False
        return True

    def get_previous_bet(self, p_index):
        for p in range(p_index - 1, p_index - self.num_players, -1):
            pm = self._mod(p)
            if not self.players[pm].folded():
                return self.players[pm].get_bet_amount()
        return 0

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
            equal_bets = self.check_equal_bets()
            if equal_bets:
                if not self.check:
                    self.finish_round()
                    continue
                opener = self.next_not_fold(self.dealer)
                if opener is not None and self.players[opener].bet_amount != 0:
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
            raise ValueError(f"Unknown action: {action!r}")
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

        end_of_hand = (
            self.winner is not None
            or self.round == 3
            or all(self.players[p].money == 0 for p in self.players_playing)
        )

        if end_of_hand:
            if self.round != 3 and self.winner is None:
                self.deal_com_cards(5 - len(self.com_cards.cards))
            self._distribute_pot()
            self.round = 0
        else:
            self.round += 1
            self.start_round()

    def collect_data(self):
        # Observation is rotated so the acting player is at index 0; positions
        # 1..n-1 are the players who act after them, in turn order. The acting
        # seat is implicit, so the old turn one-hot is replaced by a
        # dealer-relative one-hot (same dim: 104 + 3n).
        p: Player = self.players[self.turn]
        h = p.hand.to_array().ravel()
        c = self.com_cards.to_array().ravel()
        order = [self._mod(self.turn + i) for i in range(self.num_players)]
        m = np.array([self.players[i].money / self.players[i]._init_money
                      if self.players[i]._init_money > 0 else 0.0 for i in order])
        b = np.array([self.players[i].bet_amount / self.players[i]._init_money
                      if self.players[i]._init_money > 0 else 0.0 for i in order])
        d = np.zeros((self.num_players,))
        d[self._mod(self.dealer - self.turn)] = 1
        data = np.concatenate((h, c, m, b, d))
        return data

    def observe(self):
        prev = self.players[self._mod(self.turn - 1)]
        reward = prev.instant_change
        prev.instant_change = 0
        self.check_turn()
        w = -1 if self.winner is None else self.winner
        return self.collect_data(), reward, w

    def step(self, action, show=False):
        if self.winner is not None:
            raise RuntimeError("Tried to step after game won")

        if action[0] == 0:
            if show:
                print(self.current_player().name + ' called')
            self.take_turn('c')
        elif action[0] == 1:
            min_raise = self.small_blind * 2
            r = str(max(min_raise, math.ceil(action[1] * self.players[self.turn].money)))
            if show:
                print(self.current_player().name + ' raised ' + r)
            self.take_turn('r' + r)
        elif action[0] == 2:
            if show:
                print(self.current_player().name + ' folded')
            self.take_turn('f')
        else:
            raise ValueError(f"Unknown action type: {action[0]!r}")

        return self.observe()

    def reset_for_next_hand(self):
        """Advance to the next hand: rotate dealer, then reset per-hand state."""
        self.dealer = self._mod(self.dealer + 1)
        return self.reset()

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
            except ValueError:
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
