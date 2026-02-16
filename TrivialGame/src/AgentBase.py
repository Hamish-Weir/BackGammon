import inspect
from abc import ABC, abstractmethod
from src.Board import Board

class AgentBase(ABC):
    @abstractmethod
    def __init__(self, player):
        assert player == 1 or player == -1
        self.player = player

    @abstractmethod
    def make_move(self, board: Board, opp_move):
        """Makes a move based on the current board state."""
        pass

    def __hash__(self) -> int:
        return hash(inspect.getsource(self.__class__))
