import numpy as np
import regex as re


class Deck:
    def __init__(self, empty=None):
        if empty is None:
            self.cards = np.ones((4, 13))
        else:
            self.cards = np.zeros((4, 13))

    def _encode_card(self, card):
        match = re.search(r'([2-9]|10|[AJQK])[CDHS]', card)
        if match and len(match.group()) == len(card):
            index = []
            index.append(['C', 'D', 'H', 'S'].index(card[-1]))
            num = card[:len(card)]
            a = {'A': 0, 'J': 10, 'Q': 11, 'K': 12}
            try:
                index.append(int(num) - 1)
            except ValueError:
                index.append(a[num[0]])
            return index

    def _decode_card_separate(self, suit, num):
        name = ''
        if num == 0:
            name += 'A'
        elif num < 10:
            name += str(num + 1)
        else:
            ar = ['J', 'Q', 'K']
            name += str(ar[num - 10])

        suits = ['C', 'D', 'H', 'S']
        name += suits[suit]
        return name

    def _decode_card(self, index):
        return self._decode_card_separate(index[0], index[1])

    def pop(self, card):
        if type(card) == list:
            if self.cards[card[0], card[1]] == 1:
                self.cards[card[0], card[1]] = 0
                return card
            else:
                return False
        else:
            if self.cards[c[0], c[1]] == 1:
                c = self._encode_card(card)
                self.cards[c[0], c[1]] = 0
                return card
            else:
                return False

    def check_card(self, card):
        if card


class Player:
    id = 0
    def __init__(self):
        global id
        self.hand = Deck(empty=True)
        self.id = id
        id += 1

    def deal(self, card):
        assert type(card) == list
        if


class Ponk:
    def __init__(self, num_players, sb):


deck = Deck()
card = deck._encode_card('AS')
print(card)
print(deck._decode_card(card))
print(deck.pop(card))
print(deck.cards)
