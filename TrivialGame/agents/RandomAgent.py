from random import choice
from AgentBase import AgentBase
from Board import Board

class RandomAgent(AgentBase):
    def __init__(self, player):
        super().__init__(player)
        self.choice = choice

    def make_move(self, board: Board, die1: int, die2: int, opp_move):
        """Makes a move based on the current board state."""

        legal = board.get_legal_movesequences(die1,die2,self.player)
        movesequence = self.choice(legal)
        
        return movesequence