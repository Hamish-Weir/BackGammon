from random import choice
from src.AgentBase import AgentBase
from src.Board import Board

class RandomAgent(AgentBase):
    def __init__(self, player):
        super().__init__(player)
        self.choice = choice

    def make_move(self, board: Board, opp_move):
        """Makes a move based on the current board state."""
        
        legal = board.get_legal_movesequences(self.player)
        movesequence = self.choice(legal)
        
        return movesequence