"""Edge-case tests for poker gameplay in pokerbot.env.

Most tests poke at the game state directly rather than driving it through a full
hand, so that scenarios involving specific cards or all-in patterns are
deterministic.
"""

import pytest

from pokerbot.env import Card, Hand, Player, Ponk, PonkConfig


# ---------- helpers ----------

def make_game(num_players=4, starting_money=100, small_blind=1):
    return Ponk(PonkConfig(
        num_players=num_players,
        small_blind=small_blind,
        starting_money=starting_money,
    ))


def make_card(s):
    val = '10' if s[0] == 'T' else s[0]
    return Card(val, s[1].upper())


def set_hand(player, cards):
    player.hand = Hand()
    for c in cards:
        player.hand.add_card(make_card(c))


def set_community(game, cards):
    game.com_cards = Hand()
    for c in cards:
        game.com_cards.add_card(make_card(c))


def force_state(game, *, round_idx, turn, start_turn, check, pot, winner=None):
    game.round = round_idx
    game.turn = turn
    game.start_turn = start_turn
    game.check = check
    game.pot = pot
    game.winner = winner
    game.rotations = 0


# ---------- Player-level edge cases ----------

class TestPlayerBetting:
    def test_raise_more_than_money_goes_all_in(self):
        p = Player("p", money=50)
        p.raise_bet(100)
        assert p.bet_amount == 50
        assert p.money == 0
        assert p.hand_contribution == 50

    def test_call_to_more_than_money_goes_all_in(self):
        p = Player("p", money=20)
        p.call_to(50)
        assert p.bet_amount == 20
        assert p.money == 0

    def test_negative_raise_is_clamped_to_zero(self):
        p = Player("p", money=50)
        p.raise_bet(-5)
        assert p.bet_amount == 0
        assert p.money == 50


# ---------- check_equal_bets ----------

class TestCheckEqualBets:
    def test_all_matched_returns_true(self):
        game = make_game(num_players=3)
        game.reset()
        for p in game.players:
            p.bet_amount = 10
            p.money = 50
        assert game.check_equal_bets() is True

    def test_unmatched_bet_returns_false(self):
        game = make_game(num_players=3)
        game.reset()
        game.players[0].bet_amount = 10
        game.players[0].money = 50
        game.players[1].bet_amount = 5
        game.players[1].money = 50
        game.players[2].bet_amount = 10
        game.players[2].money = 50
        assert game.check_equal_bets() is False

    def test_short_stack_all_in_for_less_does_not_block(self):
        # P0 all-in for less than the bet. Others can still close the round.
        game = make_game(num_players=3)
        game.reset()
        game.players[0].bet_amount = 30
        game.players[0].money = 0  # all-in
        game.players[1].bet_amount = 50
        game.players[1].money = 50
        game.players[2].bet_amount = 50
        game.players[2].money = 50
        assert game.check_equal_bets() is True

    def test_all_in_raise_must_be_called(self):
        # Regression: an all-in raise above others' bets must NOT report equal.
        game = make_game(num_players=3)
        game.reset()
        game.players[0].bet_amount = 100  # all-in raise
        game.players[0].money = 0
        game.players[1].bet_amount = 50
        game.players[1].money = 50
        game.players[2].bet_amount = 50
        game.players[2].money = 50
        assert game.check_equal_bets() is False

    def test_everyone_all_in_returns_true(self):
        game = make_game(num_players=3)
        game.reset()
        for p in game.players:
            p.bet_amount = 30
            p.money = 0
        assert game.check_equal_bets() is True


# ---------- All-in flow through step() ----------

class TestAllInFlow:
    def _setup_river_heads_up(self, game):
        """Force a river heads-up state between P1 and P3, no bets yet."""
        game.fold(0)
        game.fold(2)
        for p in game.players:
            p.bet_amount = 0
        for i in (1, 3):
            game.players[i].money = 68
            game.players[i].hand_contribution = 32
        force_state(game, round_idx=3, turn=1, start_turn=1, check=True, pot=64)

    def test_all_in_raise_does_not_skip_opponent(self):
        """If P1 shoves on the river, P3 must get a turn — not auto-folded."""
        game = make_game(num_players=4, starting_money=100, small_blind=1)
        game.reset()
        self._setup_river_heads_up(game)

        _, _, winner = game.step([1, 1.0])

        assert winner == -1, "hand ended before P3 could respond to the all-in"
        assert game.turn == 3
        assert game.players[1].bet_amount == 68
        assert game.players[1].money == 0

    def test_all_in_then_fold_awards_pot_to_shover(self):
        game = make_game(num_players=4, starting_money=100, small_blind=1)
        game.reset()
        self._setup_river_heads_up(game)

        game.step([1, 1.0])           # P1 all-in for 68
        _, _, winner = game.step([2, 0])  # P3 folds

        assert winner == 1
        # P1 collects: pot (64) + their own bet returned (68) = 132 added to 0 stack.
        assert game.players[1].money == 132
        assert game.pot == 0


# ---------- Side pot distribution ----------

class TestSidePots:
    def _three_way_showdown(self, game):
        """P0 straight (best), P2 trips, P1 pair (worst)."""
        set_community(game, ['2c', '3d', '4h', '5s', '9c'])
        set_hand(game.players[0], ['6c', '7d'])
        set_hand(game.players[1], ['Kh', 'Kd'])
        set_hand(game.players[2], ['2d', '2h'])

    def test_short_stack_best_hand_wins_main_only(self):
        game = make_game(num_players=3, starting_money=100)
        game.reset()
        self._three_way_showdown(game)
        # P0 all-in for 50; P1 and P2 each in for 100.
        game.players[0].hand_contribution = 50
        game.players[0].money = 0
        game.players[1].hand_contribution = 100
        game.players[1].money = 0
        game.players[2].hand_contribution = 100
        game.players[2].money = 0
        for p in game.players:
            p.bet_amount = 0
        game.pot = 250

        game._distribute_pot()

        # Main pot: 50 * 3 = 150 → P0 (straight)
        # Side pot: 50 * 2 = 100 → P2 (trips beat pair)
        assert game.players[0].money == 150
        assert game.players[1].money == 0
        assert game.players[2].money == 100
        assert game.winner == 0
        assert game.pot == 0

    def test_short_stack_worst_hand_loses_everything(self):
        game = make_game(num_players=3, starting_money=100)
        game.reset()
        self._three_way_showdown(game)
        # Same hands, but P1 (pair) is the short stack now.
        game.players[0].hand_contribution = 100
        game.players[0].money = 0
        game.players[1].hand_contribution = 30
        game.players[1].money = 0
        game.players[2].hand_contribution = 100
        game.players[2].money = 0
        for p in game.players:
            p.bet_amount = 0
        game.pot = 230

        game._distribute_pot()

        # P0 (straight) wins both pots over P1/P2.
        assert game.players[0].money == 230
        assert game.players[1].money == 0
        assert game.players[2].money == 0
        assert game.winner == 0

    def test_three_contribution_levels_split_correctly(self):
        """P0=20, P1=60, P2=60. Best hands by player: P0 > P1 > P2."""
        game = make_game(num_players=3, starting_money=100)
        game.reset()
        self._three_way_showdown(game)  # P0 straight, P1 pair, P2 trips
        # Re-set so P1 (pair) is ABOVE P2 (trips) — swap their hands.
        set_hand(game.players[1], ['2d', '2h'])  # trips
        set_hand(game.players[2], ['Kh', 'Kd'])  # pair (worst)

        game.players[0].hand_contribution = 20
        game.players[0].money = 0
        game.players[1].hand_contribution = 60
        game.players[1].money = 0
        game.players[2].hand_contribution = 60
        game.players[2].money = 0
        for p in game.players:
            p.bet_amount = 0
        game.pot = 140

        game._distribute_pot()

        # Main pot (level 20): 20*3 = 60 → P0 (straight)
        # Side pot (level 60): 40*2 = 80 → P1 (trips beats pair)
        assert game.players[0].money == 60
        assert game.players[1].money == 80
        assert game.players[2].money == 0
        assert game.winner == 0


# ---------- Chops ----------

class TestChops:
    def test_two_way_chop_when_board_plays(self):
        game = make_game(num_players=2, starting_money=100)
        game.reset()
        set_community(game, ['Ac', 'Kd', 'Qh', 'Js', 'Td'])  # broadway
        set_hand(game.players[0], ['2h', '3d'])
        set_hand(game.players[1], ['4c', '5s'])
        for p in game.players:
            p.hand_contribution = 50
            p.money = 0
            p.bet_amount = 0
        game.pot = 100

        game._distribute_pot()

        assert game.players[0].money == 50
        assert game.players[1].money == 50

    def test_chop_remainder_goes_to_first_winner(self):
        """4 contributors, 3-way chop: 100 → 34/33/33/0."""
        game = make_game(num_players=4, starting_money=100)
        game.reset()
        # Board pair of 8s; top 3 players all play AK kickers, P3 plays QJ.
        set_community(game, ['8c', '8d', '4h', '5s', '2c'])
        set_hand(game.players[0], ['Ac', 'Kh'])
        set_hand(game.players[1], ['Ad', 'Ks'])
        set_hand(game.players[2], ['As', 'Kd'])
        set_hand(game.players[3], ['Qc', 'Jd'])  # weaker kickers — loses
        for p in game.players:
            p.hand_contribution = 25
            p.money = 0
            p.bet_amount = 0
        game.pot = 100

        game._distribute_pot()

        total = sum(p.money for p in game.players)
        assert total == 100
        assert game.players[3].money == 0
        winnings = sorted(
            [game.players[i].money for i in range(3)], reverse=True
        )
        assert winnings == [34, 33, 33]


# ---------- Lone non-folded player ----------

class TestSinglePlayerLeft:
    def test_pot_awarded_without_showdown(self):
        game = make_game(num_players=3, starting_money=100)
        game.reset()
        game.fold(0)
        game.fold(2)
        before = game.players[1].money
        game.pot = 75

        game._distribute_pot()

        assert game.players[1].money == before + 75
        assert game.pot == 0
        assert game.winner == 1

    def test_everyone_folds_via_play_flow(self):
        """End-to-end: pre-flop, three of four players fold; the last wins."""
        game = make_game(num_players=4, starting_money=100, small_blind=1)
        game.reset()
        # turn starts at player 3 (dealer+3 with dealer=0)
        # Sequence: 3 folds, 0 folds, 1 folds → player 2 (BB) wins
        game.step([2, 0])  # P3 fold
        game.step([2, 0])  # P0 fold
        _, _, winner = game.step([2, 0])  # P1 fold
        assert winner == 2
        # P2 should win the pot (1 SB + 2 BB = 3 chips contributed by blinds);
        # they paid 2, so net gain over starting money is +1.
        assert game.players[2].money == 100 + 1


# ---------- reset_for_next_hand ----------

class TestHandAdvancement:
    def test_dealer_rotates_and_hand_num_increments(self):
        game = make_game(num_players=4, starting_money=100)
        game.reset()
        assert game.hand_num == 1
        assert game.dealer == 0

        game.reset_for_next_hand()

        assert game.hand_num == 2
        assert game.dealer == 1
        assert len(game.com_cards.cards) == 0
        # fresh hands dealt
        for p in game.players:
            assert len(p.hand.cards) == 2

    def test_players_preserved_across_reset(self):
        game = make_game(num_players=4, starting_money=100)
        game.reset()
        ids = [id(p) for p in game.players]
        game.reset_for_next_hand()
        assert [id(p) for p in game.players] == ids
