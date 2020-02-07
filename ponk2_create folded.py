from random import randint, choice
import itertools
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


class Hand(Deck):
    def __init__(self):
        self.cards = []

class Player:
    __newid = 0
    def __init__(self, name, money=None):
        self.hand = Hand()
        self.name = name
        self.id = Player.__newid
        Player.__newid += 1
        self.money = 0
        self.bet_amount = 0
        if money is not None:
            self.money = int(money)

    def deal(self, card):
        self.hand.add_card(card)

    def bet(self, amount):
        if amount <= self.money and amount > 0:
            self.money -= amount - self.bet_amount
            self.bet_amount = amount
        else:
            self.bet_amount = self.money
            self.money = 0

    def get_name(self):
        return self.name

    def get_bet_amount(self):
        return self.bet_amount


class Ponk:
    def __init__(self, smb):
        self.players = []
        self.folded_players = []
        self.dealer_id = 0
        self.turn = 0
        self.start_turn = 0
        self.pot = 0
        self.deck = Deck()
        self.smb = smb
        self.com_cards = Hand()
        self.round = 0
        self.num_players = 0
        self.folded_num = 0
        self.winner = -1

    def add_player(self,money,name):
        self.players.append(Player(name,money=money))
        self.num_players += 1

    def get_turn_name(self):
        return self.players[self.turn].get_name()

    def get_turn_id(self):
        return self.turn

    def _deal_hands(self):
        for i in range(len(self.players)):
            self.players[i].deal(self.deck.pop_random())
            self.players[i].deal(self.deck.pop_random())

    def step_turn(self):
        self.turn = self.mod(self.turn + 1)

    def fold(self, p_index):
        self.num_players -= 1
        if self.num_players == 1:
            self.win(p_index=0)
        self.folded_ids.append(self.players[p_index].id)
        self.folded_players.append(self.players.pop(p_index))
        self.folded_num += 1

    def bet(self, p_index, amount):
        self.players[p_index].bet(amount)

    def deal_com_cards(self, n):
        for _ in range(n):
            self.com_cards.add_card(self.deck.pop_random())

    def mod(self,n):
        return n % self.num_players

    def is_playing(self,id):
        return any(p.id == id for p in self.players)

    def get_start_turn(self):
        if

    def win(self, p_index=None):
        if p_index is None:
            self.current_player().money += self.pot
            self.winner = self.current_player().id
            print(self.get_turn_name() + " has won!")
        else:
            self.players[p_index].money += self.pot
            self.winner = self.players[p_index].id
            print(self.players[p_index].name + " has won!")
        self.pot = 0

    def current_player(self):
        return self.players[self.turn]

    def show_player(self, p_index):
        return [self.players[p_index].name + " " +
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

    def check_equal_bets(self):
        if all(p.bet_amount == self.players[0].bet_amount for p in self.players):
            return True
        return False

    def get_previous_bet(self, p_index):
        return self.players[self.mod(p_index - 1)].bet_amount

    def finish_round(self):
        tot = 0
        for p in self.players:
            tot += p.bet_amount
            p.bet_amount = 0
        self.pot = tot

    def game_setup(self):
        self._deal_hands()
        self.bet(self.mod(self.dealer_id + 1), self.smb)
        self.bet(self.mod(self.dealer_id + 2), self.smb * 2)
        self.start_turn = self.mod(self.dealer_id + 3)

    def _start_round(self):
        if self.round == 0:
            self.game_setup()
        elif self.round == 1:
            self.start_turn = self
            self.turn = self.mod(self.start_turn)
        elif self.round == 2 or self.round == 3:
            self.turn = self.mod(self.start_turn)
            self.deal_com_cards(1)
        check = False
        rotations = 0
        while self.winner == -1:
            if self.turn == self.start_turn:
                rotations += 1
            if self.check_equal_bets() and self.players[self.start_turn].bet_amount != 0:
                break
            print("turn: "+ str(self.turn))
            self.print_players()

            crf = input(self.get_turn_name() + ": Call, Raise or Fold? (c/r/f)")
            if crf == 'f':
                self.fold(self.turn)
                if self.turn == self.start_turn:
                    self.start_turn = self.mod(self.start_turn + 1)
                self.folded_num += 1
            elif crf == 'c':
                bet = self.get_previous_bet(self.turn)
                if self.turn == self.start_turn and rotations == 1 and bet == 0:
                    check = True
                self.bet(self.turn, bet)
            elif crf == 'r':
                if check:
                    check = False
                r = int(input("Input raise amount for " + self.get_turn_name()))
                self.bet(self.turn, r + self.get_previous_bet(self.turn))
            else:
                raise Exception
            self.step_turn()
        self.finish_round()
        self.round += 1

    def play(self, dealer = None):
        if dealer:
            self.dealer_id = dealer
        for i in range(4):
            print("Round ", self.round)
            self._start_round()



ponker = Ponk(1)
ponker.add_player(1000,"Alice")
ponker.add_player(1000,"Bob")
ponker.add_player(1000,"Clarisse")

for i in ponker.current_player_show():
    print(i)
ponker.play(dealer = 0)
