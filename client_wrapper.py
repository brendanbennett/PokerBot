import ponk2

class Interface:
    def __init__(self):
        self.game = ponk2.ponker()

    def observe_current_player(self):
        p_id = self.game.get_turn_id()
        p = self.game.players[p_id]
        money = p.get_money
